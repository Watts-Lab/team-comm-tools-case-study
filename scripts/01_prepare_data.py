"""Step 1 - turn the raw experiment pickles into two tidy CSVs per split.

Outputs (per split, into data/processed/):
  * ``chat_{split}.csv``  - one row per message, in the four-column format the
    Team Communication Toolkit expects: conversation id, speaker id, message,
    timestamp. Extra bookkeeping columns ride along but are ignored by TCT.
  * ``games_{split}.csv`` - one row per game: the outcome variable, whether the
    group could talk, and the design parameters of the game.
  * ``rounds_{split}.csv`` - one row per game per round, used only to plot how
    contribution evolves over the course of a game.

Run:  python scripts/01_prepare_data.py
"""

import json
import pickle
import sys

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

    out = (rounds
           .merge(games, on="gameId", how="left")
           .merge(treatments, on="treatmentId", how="left")
           .merge(configs, left_on="treatment_name",
                  right_on="CONFIG_treatmentName", how="left"))
    return out


# --------------------------------------------------------- game outcomes -----
def build_game_table(rounds):
    """Collapse player-rounds to one row per game.

    The outcome is the group's mean contribution *rate* in the penultimate round.
    The penultimate round is used rather than the last because groups that know
    the game is ending routinely defect in the final round, which says more about
    the horizon than about how the group got along.
    """
    rounds = rounds.dropna(subset=["data.contribution"]).copy()
    rounds["contribution_rate"] = rounds["data.contribution"] / ENDOWMENT

    last_round = rounds.groupby("gameId")["round_index"].max().rename("last_round")
    n_rounds_played = (rounds.groupby("gameId")["round_index"].nunique()
                       .rename("n_rounds_played"))

    game = pd.concat([last_round, n_rounds_played], axis=1).reset_index()
    game = game[game["n_rounds_played"] >= MIN_ROUNDS].copy()
    game["penultimate_round"] = game["last_round"] - 1

    # Outcome: mean contribution rate in the penultimate round.
    pen = rounds.merge(game[["gameId", "penultimate_round"]], on="gameId")
    pen = pen[pen["round_index"] == pen["penultimate_round"]]
    game = game.merge(
        pen.groupby("gameId")["contribution_rate"].mean()
           .rename("contribution_penultimate").reset_index(),
        on="gameId", how="inner")

    # Secondary outcome, for robustness checks: mean rate across all rounds.
    game = game.merge(
        rounds.groupby("gameId")["contribution_rate"].mean()
              .rename("contribution_all_rounds").reset_index(),
        on="gameId", how="left")

    # Design parameters are constant within a game.
    config_cols = [c for c in ["CONFIG_chat"] + CONFIG_COLS if c in rounds.columns]
    game = game.merge(rounds.groupby("gameId")[config_cols].first().reset_index(),
                      on="gameId", how="left")
    game["has_chat_channel"] = game["CONFIG_chat"].astype(bool)
    return game


def build_round_table(rounds, game):
    """Mean contribution rate per game per round, for the trajectory figure."""
    rounds = rounds.dropna(subset=["data.contribution"]).copy()
    rounds["contribution_rate"] = rounds["data.contribution"] / ENDOWMENT
    per_round = (rounds.groupby(["gameId", "round_index"])["contribution_rate"]
                 .mean().rename("contribution_rate").reset_index())
    return per_round.merge(
        game[["gameId", "has_chat_channel", "last_round", "n_rounds_played"]],
        on="gameId", how="inner")


def filter_chat_to_predictive_window(chat, game):
    """Keep only messages a group could have sent *before* the outcome is decided.

    Contributions for round ``k`` are made during that round's "contribution"
    phase, so everything from rounds before the penultimate round is fair game,
    plus the penultimate round's own contribution-phase chat. Anything from the
    penultimate round's outcome/summary phases onward would leak the result.
    """
    chat = chat.merge(game[["gameId", "penultimate_round"]], on="gameId", how="inner")
    before = chat["round_index"] < chat["penultimate_round"]
    during = ((chat["round_index"] == chat["penultimate_round"])
              & (chat["phase"] == "contribution"))
    return chat[before | during].drop(columns=["penultimate_round"])


# ------------------------------------------------------------------ main -----
def prepare(split):
    print(f"\n=== {split} ===")
    md = load_master(split)

    rounds = attach_configs(md.df_rounds.copy(), md)
    game = build_game_table(rounds)

    per_round = build_round_table(rounds, game)

    chat = parse_chat_log(md.df_games)
    chat = chat.dropna(subset=["timestamp"])
    chat = chat[chat["text"].astype(str).str.strip() != ""]
    chat = filter_chat_to_predictive_window(chat, game)
    chat = chat.sort_values(["gameId", "timestamp"]).reset_index(drop=True)

    # A group only counts as "communicating" if the channel was open *and* it was
    # actually used; a chat-enabled group that never typed has no conversation to
    # extract features from.
    talked = set(chat["gameId"].unique())
    game["did_communicate"] = game["gameId"].isin(talked)

    per_round.to_csv(DATA_PROCESSED / f"rounds_{split}.csv", index=False)

    chat_path = DATA_PROCESSED / f"chat_{split}.csv"
    game_path = DATA_PROCESSED / f"games_{split}.csv"
    chat.to_csv(chat_path, index=False)
    game.to_csv(game_path, index=False)

    print(f"games: {len(game)}  (channel open: {int(game.has_chat_channel.sum())}, "
          f"actually talked: {int(game.did_communicate.sum())})")
    print(f"messages kept: {len(chat)} across {chat.gameId.nunique()} games")
    print(f"mean penultimate contribution rate: "
          f"{game.contribution_penultimate.mean():.3f}")
    print(f"wrote {chat_path.name}, {game_path.name}")


if __name__ == "__main__":
    for split in SPLITS:
        prepare(split)
