"""Step 1 - turn the raw experiment pickles into tidy CSVs per split.

The unit of analysis is a **game-round**: one group, one round of play. That is
what makes the question answerable - there are only a few hundred games, but
several thousand game-rounds.

Prediction runs strictly forward in time. A round's conversation is used to
predict the contribution decision in the *next* round:

    talk during round k   ->   contribution in round k+1

Round k's chat spans its contribution, outcome, and summary phases, so it
includes the group reacting to how round k turned out. Using it to predict round
k+1 keeps every message strictly earlier than the decision it predicts, which
rules out leakage without discarding most of the text.

Outputs (per split, into data/processed/):
  * ``chat_{split}.csv``   - one row per message, in the four-column format the
    Team Communication Toolkit expects, plus a ``conv_id`` naming the
    game-round-and-block it belongs to and the ``target_round`` it predicts.
  * ``rounds_{split}.csv`` - one row per game-round: the outcome, the previous
    round's contribution, whether the group had a channel, the design parameters,
    and where the round sits in the game under both staging schemes.

Run:  python scripts/01_prepare_data.py
"""

import json
import pickle
import sys

import numpy as np
import pandas as pd

from config import (COMPAT, CONFIG_COLS, DATA_PROCESSED, ENDOWMENT, MIN_ROUNDS,
                    RAW_PKL, SPLITS)

sys.path.insert(0, str(COMPAT))  # makes `pgg_helper` importable for pickle


# --------------------------------------------------------------- loading -----
def load_master(split):
    """Unpickle one split's master-data object."""
    with open(RAW_PKL[split], "rb") as f:
        return pickle.load(f)


def parse_chat_log(df_games):
    """Flatten the JSON chat log stored on each game row into one row per message.

    ``chat_log`` holds comma-separated JSON objects without the enclosing
    brackets, so they are re-wrapped before parsing.
    """
    # `itertuples` mangles column names that are not valid identifiers (``_id``
    # becomes ``_3``), so select and rename the two columns we need up front.
    games = df_games[["_id", "chat_log"]].rename(columns={"_id": "gameId"})

    records = []
    for game_id, log in games.itertuples(index=False):
        if not isinstance(log, str) or not log.strip():
            continue
        try:
            messages = json.loads("[" + log + "]")
        except json.JSONDecodeError:
            print(f"  ! could not parse chat_log for game {game_id}")
            continue
        for msg in messages:
            phase = msg.get("gamePhase", "")
            # Phases look like "Round 3 - contribution"; pull out the two parts.
            round_index, phase_name = -1, ""
            if phase.startswith("Round "):
                head, _, tail = phase.partition(" - ")
                phase_name = tail.strip()
                try:
                    round_index = int(head.split(" ")[1])
                except (IndexError, ValueError):
                    pass
            records.append({
                "gameId": game_id,
                "playerId": msg.get("playerId", ""),
                "avatar": msg.get("avatar", ""),
                "text": msg.get("text", ""),
                "timestamp": msg.get("timestamp", ""),
                "round_index": round_index,
                "phase": phase_name,
            })

    chat = pd.DataFrame.from_records(records)
    chat["timestamp"] = pd.to_datetime(chat["timestamp"], errors="coerce", utc=True)
    return chat


def attach_configs(rounds, md):
    """Join each game's design parameters onto the player-round table.

    Join path: rounds -> games (treatmentId) -> treatments (name) -> configs.
    """
    games = md.df_games[["_id", "treatmentId"]].rename(columns={"_id": "gameId"})
    treatments = md.df_treatments[["_id", "name"]].rename(
        columns={"_id": "treatmentId", "name": "treatment_name"})
    configs = md.df_treatment_config

    for df in (rounds, games, treatments):
        for col in ("gameId", "treatmentId"):
            if col in df.columns:
                df[col] = df[col].astype(str)

    return (rounds
            .merge(games, on="gameId", how="left")
            .merge(treatments, on="treatmentId", how="left")
            .merge(configs, left_on="treatment_name",
                   right_on="CONFIG_treatmentName", how="left"))


# ---------------------------------------------------------- game-rounds -----
def build_round_table(rounds):
    """Collapse player-rounds to one row per game-round.

    Carries two pieces of timing information that matter in a repeated game and
    are known before any talk happens: how far into the game the round sits, and
    how many rounds are left. Groups defect predictably as the end approaches, so
    a model without these would credit that to whatever was being said at the time.
    """
    rounds = rounds.dropna(subset=["data.contribution"]).copy()
    rounds["contribution_rate"] = rounds["data.contribution"] / ENDOWMENT

    game_rounds = (rounds.groupby(["gameId", "round_index"])
                   .agg(contribution_rate=("contribution_rate", "mean"),
                        n_players_active=("playerId", "nunique"))
                   .reset_index())

    played = game_rounds.groupby("gameId")["round_index"].agg(["max", "nunique"])
    played.columns = ["last_round", "n_rounds_played"]
    game_rounds = game_rounds.merge(played, on="gameId")
    game_rounds = game_rounds[game_rounds["n_rounds_played"] >= MIN_ROUNDS]

    game_rounds["rounds_remaining"] = game_rounds["last_round"] - game_rounds["round_index"]
    game_rounds["round_position"] = (game_rounds["round_index"]
                                     / game_rounds["last_round"].clip(lower=1))
    game_rounds["is_last_round"] = game_rounds["rounds_remaining"] == 0

    config_cols = [c for c in ["CONFIG_chat"] + CONFIG_COLS if c in rounds.columns]
    game_rounds = game_rounds.merge(
        rounds.groupby("gameId")[config_cols].first().reset_index(),
        on="gameId", how="left")
    game_rounds["has_chat_channel"] = game_rounds["CONFIG_chat"].astype(bool)

    # What the group did last round. Contributions are strongly autocorrelated and
    # POST-block talk is a reaction to this number, so it has to be a control
    # rather than left for the conversation features to proxy for.
    game_rounds = game_rounds.sort_values(["gameId", "round_index"])
    game_rounds["lagged_contribution"] = (game_rounds.groupby("gameId")
                                          ["contribution_rate"].shift(1))

    # Round 0 is kept. It has no previous round, so no POST block and no lagged
    # contribution - its POST features fall back to the neutral value, which is
    # exactly how a round with a channel but no reaction talk is already handled.
    game_rounds["stage_absolute"] = np.where(
        game_rounds["rounds_remaining"] <= 2, "endgame",
        np.where(game_rounds["round_index"] <= 2, "opening", "middle"))
    game_rounds["stage_relative"] = pd.cut(
        game_rounds["round_position"], [0, 1 / 3, 2 / 3, 1.01],
        right=False, labels=["early", "middle", "late"]).astype(str)
    game_rounds["conv_id_pre"] = (game_rounds["gameId"] + "_r"
                                  + game_rounds["round_index"].astype(str) + "_pre")
    game_rounds["conv_id_post"] = (game_rounds["gameId"] + "_r"
                                   + (game_rounds["round_index"] - 1).astype(str) + "_post")
    return game_rounds


def label_chat_with_block(chat, game_rounds):
    """Assign each message to the PRE or POST block, and name its conversation.

    A message's conversation id encodes the round it was spoken in and its block,
    so the toolkit treats deliberation and reaction as separate conversations.
    """
    chat = chat.copy()
    chat["block"] = chat["phase"].map({"contribution": "pre",
                                       "outcome": "post", "summary": "post"})
    chat = chat.dropna(subset=["block"])
    chat["conv_id"] = (chat["gameId"] + "_r" + chat["round_index"].astype(str)
                       + "_" + chat["block"])

    # PRE talk predicts its own round; POST talk predicts the round after it.
    chat["target_round"] = chat["round_index"] + (chat["block"] == "post").astype(int)

    wanted = set(game_rounds["conv_id_pre"]) | set(game_rounds["conv_id_post"])
    return chat[chat["conv_id"].isin(wanted)].copy()


# ------------------------------------------------------------------ main -----
def prepare(split):
    print(f"\n=== {split} ===")
    md = load_master(split)

    rounds = attach_configs(md.df_rounds.copy(), md)
    game_rounds = build_round_table(rounds)

    chat = parse_chat_log(md.df_games)
    chat = chat.dropna(subset=["timestamp"])
    chat = chat[chat["text"].astype(str).str.strip() != ""]
    chat = label_chat_with_block(chat, game_rounds)
    chat = chat.sort_values(["conv_id", "timestamp"]).reset_index(drop=True)

    spoke = set(chat["conv_id"].unique())
    game_rounds["had_pre_talk"] = game_rounds["conv_id_pre"].isin(spoke)
    game_rounds["had_post_talk"] = game_rounds["conv_id_post"].isin(spoke)
    game_rounds["had_conversation"] = (game_rounds["had_pre_talk"]
                                       | game_rounds["had_post_talk"])

    chat_path = DATA_PROCESSED / f"chat_{split}.csv"
    rounds_path = DATA_PROCESSED / f"rounds_{split}.csv"
    chat.to_csv(chat_path, index=False)
    game_rounds.to_csv(rounds_path, index=False)

    print(f"game-rounds: {len(game_rounds)} from {game_rounds.gameId.nunique()} games")
    print(f"  channel open: {int(game_rounds.has_chat_channel.sum())}")
    print(f"  by absolute stage: {game_rounds.stage_absolute.value_counts().to_dict()}")
    print(f"  with PRE (deliberation) talk:  {int(game_rounds.had_pre_talk.sum())}")
    print(f"  with POST (reaction) talk:     {int(game_rounds.had_post_talk.sum())}")
    print(f"messages: {len(chat)} across {chat.conv_id.nunique()} block-conversations "
          f"({chat.block.value_counts().to_dict()})")
    print(f"mean contribution rate: {game_rounds.contribution_rate.mean():.3f}")
    print(f"wrote {chat_path.name}, {rounds_path.name}")


if __name__ == "__main__":
    for split in SPLITS:
        prepare(split)
