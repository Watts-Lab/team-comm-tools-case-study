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

# Columns of the TCT conversation-level output that are identifiers or passed-through
# input columns rather than conversation features. The toolkit carries the first row
# of the input frame along for reference, so `timestamp` and `round_index` are data
# about one arbitrary message, not features of the conversation.
TCT_ID_COLS = {"conversation_num", "gameId", "conversation_id", "Unnamed: 0",
               "playerId", "avatar", "text", "timestamp", "round_index", "phase"}
