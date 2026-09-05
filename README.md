# Case study replication package

<https://github.com/Watts-Lab/team-comm-tools-case-study>

Code and outputs for the case study in *`team_comm_tools`: A Python Toolkit for
Exploring Text Conversations in Groups*. The pipeline turns two experiment
pickles from Alsobay et al. (2026) into the figures, tables and numbers the paper
reports, using [`team_comm_tools`](https://github.com/Watts-Lab/team_comm_tools)
v0.1.8 for feature extraction.

The corpus is the public goods game data collected by Alsobay, M., Watts, D. J.,
& Almaatouq, A. (2026). Integrative experiments identify how punishment affects
welfare in public goods games. *Science*, 392(6794).
<https://doi.org/10.1126/science.aeb5280>

This file explains how to run the pipeline and where each output goes. The paper
itself covers what the results mean.

## Requirements

Python 3.12 and the packages in `requirements.txt`:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

If spaCy or NLTK resources are missing, run `download_resources` once; the
toolkit installs that command.

Figures are set in Latin Modern, the typeface the paper itself uses. The four
faces are bundled in `scripts/fonts/` under the GUST Font License, so no TeX
installation is needed; `scripts/style.py` falls back to a TeX Live copy, and
then to a serif default, if they are missing.

Run every script from the repository root. Python then puts `scripts/` on the
path, which is how each step imports `config`:

```bash
python scripts/01_prepare_data.py
```

## Running the pipeline

```bash
python scripts/run_all.py            # steps 1 to 19, in order
bash scripts/status.sh -w            # progress, in a second terminal
```

Two steps dominate the wall clock. Step 2 runs the toolkit over about 25,000
messages and takes roughly an hour on a laptop CPU, and step 11 refits every
model in every cell and takes several hours. Step 4 takes about 40 minutes and
accepts `--only` to re-run one section:

```bash
python scripts/04_analysis.py --only stages
python scripts/04_analysis.py --help     # every section, with its cost
```

Steps 8 to 11 can also run unattended through `bash scripts/run_windows_chain.sh`,
tracked by `bash scripts/windows_status.sh -w`.

Step 17 stands outside `run_all.py` because it clears the embedding cache to time
a cold start. It moves the cached vectors to `outputs/vector_cache_backup/`, so a
run can be undone by moving them back.

```bash
python scripts/17_cold_init.py --split learn
```

## The steps

| Step | Script | Reads | Writes |
|---|---|---|---|
| 1 | `01_prepare_data.py` | `data/raw/*.pkl` | `data/processed/{chat,rounds}_{learn,val}.csv` |
| 2 | `02_extract_features.py` | `chat_{split}.csv` | `outputs/features/output/{chat,user,conv}/` |
| 3 | `03_build_analysis_table.py` | step 2 output | `data/processed/analysis_{split}.csv`, `outputs/tables/feature_manifest.csv` |
| 4 | `04_analysis.py` | `analysis_{split}.csv` | `outputs/tables/*.csv` and `outputs/tables/diagnostics/*.csv` (nine sections) |
| 6 | `06_feature_examples.py` | step 2 output | `outputs/examples/*.txt` |
| 7 | `07_lexicon_words.py relative` | `chat_{split}.csv` | `outputs/tables/lexicon_matches_relative.csv` |
| 8 | `08_cumulative_chat.py` | `chat_{split}.csv` | `data/processed/chat_cumulative_{split}.csv` |
| 9 | `09_extract_cumulative_features.py` | step 8 output | `outputs/features_cumulative/` |
| 10 | `10_build_windows_table.py` | steps 3 and 9 | `data/processed/analysis_windows_{split}.csv` |
| 11 | `11_window_compare.py` | step 10 output | `outputs/tables/windows/*.csv`, `diagnostics/block_clustering.csv` (five sections) |
| 13 | `13_window_figures.py` | `windows/block_delta_r2.csv`, `windows/block_feature_effects.csv` | `outputs/figures/{main,appendix}/*.png` |
| 14 | `14_descriptive_figures.py` | step 1 output | `outputs/figures/main/fig0_*.png`, `outputs/tables/channel_effect_tests.csv` |
| 15 | `15_short_game_scope.py` | `analysis_{split}.csv` | `outputs/tables/short_game_{scope,composition}.csv` |
| 16 | `16_runtime_table.py` | `outputs/logs/feature_builder.log` | `outputs/tables/runtime*.csv`, `runtime_table.tex` |
| 17 | `17_cold_init.py` | `chat_{split}.csv` | `outputs/tables/runtime_cold_init.csv` |
| 18 | `18_pre_post_power.py` | `chat_{split}.csv`, `analysis_{split}.csv` | `outputs/tables/pre_post_power.csv` |
| 19 | `19_pipeline_figure.py` | steps 3, 11 | `outputs/figures/main/fig3_analysis_pipeline.png` |

There is no step 5, and step 12 was removed along with the first-N analysis it
computed. The numbering is left as it is so that the remaining steps keep the
names they have in the logs.

## Where each result in the paper comes from

Figures, by the number each one has in the paper:

| Paper | File |
|---|---|
| Figure 3 | `outputs/figures/main/fig0_communication_effect.png` |
| Figure 4 | `outputs/figures/main/fig0_corpus.png` |
| Figure 5 | `outputs/figures/main/fig3_analysis_pipeline.png` |
| Figure 6 | `outputs/figures/main/fig1_when_talk_matters.png` |
| Figure 7 | `outputs/figures/main/fig2_effects_across_stages.png` |
| Figure E1 | `outputs/figures/appendix/figS1_when_talk_matters_talkers_only.png` |
| Figure E2 | `outputs/figures/appendix/figS2_effects_across_stages_talkers_only.png` |
| Figure E3 | `outputs/figures/appendix/figS3_when_talk_matters_random_forest.png` |
| Figure E4 | `outputs/figures/appendix/figS4_when_talk_matters_wide_windows.png` |
| Figure E5 | `outputs/figures/appendix/figS5_effects_across_stages_wide_windows.png` |

The step 13 figures each have a `_caption.txt` beside them, written by the same script.

Tables and reported numbers:

| Paper | File |
|---|---|
| Section 4.1.2, the communication effect | `outputs/tables/channel_effect_tests.csv` (step 14), `channel_effect.csv` (step 4) |
| Section 4.3, the feature counts | `outputs/tables/feature_manifest.csv`, `outputs/logs/feature_builder.log` |
| Table 3, the runtime | `outputs/tables/runtime_table.tex`, from `runtime.csv`, `runtime_steps.csv` and `runtime_cold_init.csv` |
| Section 4.5, additional variance explained | `outputs/tables/windows/block_delta_r2.csv` |
| Section 4.6, per-feature effects | `outputs/tables/windows/block_feature_effects.csv` |
| Section 4.6, the `relative` lexicon | `outputs/tables/lexicon_matches_relative.csv` |
| Section 4.5, how much talk each boundary holds | `outputs/tables/pre_post_power.csv` |
| Section 4.6, the conversations quoted | `outputs/examples/*.txt` |
| Appendix E.1 to E.4 and E.6 | `outputs/tables/windows/block_delta_r2.csv` (columns `controls`, `sample`, `model_family`, `binning`) |
| Appendix E.4, screen agreement | `outputs/tables/windows/block_agreement.csv` |
| Appendix E.5, proportional thirds | `outputs/tables/round_stage.csv` |
| Appendix E.7, short games | `outputs/tables/short_game_scope.csv`, `short_game_composition.csv` |
| Appendix E.8, the paired comparison | `outputs/tables/windows/block_paired_contrasts.csv`, `block_paired.csv` |

`block_delta_r2.csv` holds every cell of the design in one file. A row is one
combination of `binning` (stage, round_bin, all), `bin`, `block` (pre, post,
window, cumulative), `sample` (channel, talkers), `model_family` and `controls`
(rules+timing, rules+timing+momentum). The main text reports the ElasticNet on
the channel sample with `rules+timing` controls; each appendix test varies one of
those columns.

`outputs/tables/diagnostics/` holds the tables behind decisions the paper states
in prose:

| File | What it holds |
|---|---|
| `model_comparison.csv`, `variance_decomposition.csv` | each block of predictors added to the model in turn, with the R² it buys |
| `speech_vs_content.csv` | what speaking at all adds, and what the content adds on top of it, per boundary |
| `family_importance.csv`, `family_importance_opening.csv` | how much prediction is lost when one feature family is dropped, pooled and in the Opening |
| `stage_profile.csv`, `stage_agreement.csv`, `stage_examples.csv` | how each feature behaves across the three time periods, and sample messages from each |
| `feature_effects.csv` | the per-feature screen pooled over all rounds, which the by-period screens refine |
| `block_clustering.csv` | every cell of the design, whether it was fitted, and the rounds and games behind it |

## Layout

```
.
├── data/
│   ├── raw/           two experiment pickles, as collected
│   └── processed/     tidy CSVs from steps 1, 3, 8 and 10
├── scripts/
│   ├── config.py      paths, controls, feature families, the seed
│   ├── modeling.py    cross-validation, bootstrap and regression helpers
│   ├── style.py       figure conventions
│   ├── compat/        stub module so the raw pickles unpickle
│   ├── run_all.py     steps 1 to 16
│   ├── status.sh      pipeline tracker; -w to watch
│   └── 01..17         the steps in the table above
└── outputs/
    ├── features/      toolkit output for the per-round conversations
    ├── features_cumulative/   toolkit output for the cumulative conversations
    ├── tables/        every result as CSV
    │   ├── windows/      the four-boundary comparison
    │   └── diagnostics/  supporting tables the paper does not print
    ├── examples/      conversations behind each reported feature
    ├── figures/       main/ and appendix/
    └── logs/          the toolkit's run logs, read by step 16
```

Five paths are regenerable and not tracked: `outputs/features/`,
`outputs/features_cumulative/`, `outputs/vector_cache/`,
`outputs/vector_cache_backup/` and `outputs/logs/`. The
two analysis tables the window chain expands, `chat_cumulative_*.csv` and
`analysis_windows_*.csv`, are not tracked either. Everything the paper cites is
tracked, so the tables and figures can be checked without re-running anything.

Re-running step 2 or step 9 rewrites `outputs/logs/feature_builder.log`, which
step 16 parses for the runtime table.

## Reproducibility notes

`config.py` sets one seed (2139) for every model, split and bootstrap. Model
selection, scaling and hyperparameter tuning happen inside the learning split;
the validation games are scored once. Cross-validation folds hold out whole
games, and bootstrap intervals resample games.

The toolkit call in step 2 fixes three settings that the paper reports:
`drop_redundant_columns=True` with `corr_thresh=0.9`, and `treat_zero_as_na=False`
so that a zero count stays a zero.

Runtimes in Table 3 come from one machine, a 2024 MacBook Pro (M4 Pro, macOS
26.5.2). Step 16 reports whatever the log holds, so re-running the pipeline on
other hardware changes the table.
