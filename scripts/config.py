"""Shared paths and constants for the case study.

Every script imports from here so that a single edit changes the whole pipeline.
"""

from pathlib import Path

# ---------------------------------------------------------------- paths -----
ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
OUTPUTS = ROOT / "outputs"
FEATURES = OUTPUTS / "features"
# FeatureBuilder nests its own output/{chat,user,conv}/ tree under FEATURES.
CONV_FEATURES = FEATURES / "output" / "conv"
TABLES = OUTPUTS / "tables"
FIGURES = OUTPUTS / "figures"
VECTOR_CACHE = ROOT / "outputs" / "vector_cache"

for _d in (DATA_PROCESSED, FEATURES, TABLES, FIGURES):
    _d.mkdir(parents=True, exist_ok=True)

# The pickles were written from a `pgg_helper.preprocess.master_data` instance;
# scripts/compat holds a stub of that module so pickle can resolve the class.
COMPAT = Path(__file__).resolve().parent / "compat"

RAW_PKL = {
    "learn": DATA_RAW / "learning_set_master_data.pkl",
    "val": DATA_RAW / "validation_set_master_data.pkl",
}
SPLITS = ("learn", "val")

# ------------------------------------------------------------ constants -----
SEED = 2139

# Every game in this dataset gives each player the same per-round endowment, so
# contribution is rescaled to a 0-1 rate to keep games comparable.
ENDOWMENT = 20

# A game must have at least this many completed rounds to have a penultimate
# round that is distinct from the opening round.
MIN_ROUNDS = 3

# Game-level design parameters (the "configs"). These describe the *rules* of the
# game and are known before a word is spoken, so they are the natural control set:
# any predictive power that conversation features add must be on top of these.
CONFIG_COLS = [
    "CONFIG_playerCount",
    "CONFIG_numRounds",
    "CONFIG_showNRounds",
    "CONFIG_multiplier",
    "CONFIG_MPCR",
    "CONFIG_allOrNothing",
    "CONFIG_defaultContribProp",
    "CONFIG_punishmentExists",
    "CONFIG_punishmentCost",
    "CONFIG_punishmentMagnitude",
    "CONFIG_rewardExists",
    "CONFIG_rewardCost",
    "CONFIG_rewardMagnitude",
    "CONFIG_showOtherSummaries",
    "CONFIG_showPunishmentId",
    "CONFIG_showRewardId",
]

# Timing and group facts that are known before anyone speaks. They belong with the
# design parameters as controls: groups defect predictably as the end of a game
# approaches, and a model without that would credit the drop to whatever was being
# said at the time.
TIMING_COLS = [
    "round_index",
    "rounds_remaining",
    "round_position",
    "is_last_round",
    "n_players_active",
]

# Feature families. The toolkit returns 100+ columns drawn from a dozen different
# papers; grouping them by the construct family they came from turns "which of 136
# columns matters" into the answerable "which *kind* of talk matters". Order is
# meaningful: the first match wins, so the specific suffixes come before the
# catch-all prefixes.
FEATURE_FAMILIES = [
    ("Politeness",             r"_politeness_convokit$"),
    ("Receptiveness",          r"_receptiveness_yeomans$"),
    ("Lexical (LIWC)",         r"_lexical_wordcount$"),
    ("Sentiment & emotion",    r"(positive_bert|negative_bert|neutral_bert"
                               r"|textblob_|positivity_zscore|certainty_rocklage)"),
    ("Questions & repair",     r"(num_question_naive|NTRI|hedge_naive)"),
    ("Participation & timing", r"(turn_taking_index|gini_coefficient|team_burstiness"
                               r"|time_diff)"),
    ("Semantic dynamics",      r"(mimicry|forward_flow|discursive_diversity"
                               r"|info_diversity|variance_in_DD|within_person_disc_range"
                               r"|incongruent_modulation|_accommodation"
                               r"|info_exchange_zscore|first_pronouns_proportion)"),
    ("Volume & form",          r"(num_words|num_chars|num_messages|num_links"
                               r"|num_line_breaks|num_bullet_points|num_all_caps"
                               r"|num_block_quote|num_quotes|num_ellipses"
                               r"|num_numbered_points|num_emoji|num_emphasis"
                               r"|num_reddit_users|num_parentheses|word_TTR"
                               r"|dale_chall_score)"),
]

# Columns of the TCT conversation-level output that are identifiers or passed-through
# input columns rather than conversation features. The toolkit carries the first row
# of the input frame along for reference, so `timestamp` and `round_index` are data
# about one arbitrary message, not features of the conversation.
TCT_ID_COLS = {"conversation_num", "conv_id", "gameId", "conversation_id",
               "Unnamed: 0", "playerId", "avatar", "text", "timestamp",
               "round_index", "phase", "source_round", "target_round"}
