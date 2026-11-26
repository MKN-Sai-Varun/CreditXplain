#!/usr/bin/env bash
# Rebuilds every model artifact and report from the raw Home Credit files.
#
# Usage (from the project root):
#   scripts/run_pipeline.sh            run all steps
#   scripts/run_pipeline.sh 4          start at step 4 (earlier outputs must exist)
#   scripts/run_pipeline.sh --list     show the steps
#
# Needs the Kaggle files in data/home_credit/ (see README, "Data").
set -euo pipefail

cd "$(dirname "$0")/.."
PY=${PYTHON:-python3}

STEPS=(
  "Clean and merge application, bureau and previous-application data|$PY src/data_prep.py --data-path data/home_credit/ --out data/processed/app_train_clean.csv"
  "Stratified 80/20 split and baseline model comparison|$PY src/train_baseline.py"
  "Feature engineering v2 for train and test|$PY src/feature_engineering_v2.py"
  "Final LightGBM, cutoffs, model card, bands, gender ablation|$PY src/final_model.py --ablation"
  "Typical values for fields an application leaves out|$PY src/build_feature_defaults.py"
  "Plausibility reference for recourse|$PY -m src.build_recourse_reference"
  "SHAP and LIME explanations and report|$PY src/explainability.py"
  "SHAP vs LIME agreement on 150 applicants|$PY -m src.explanation_eval"
  "Offline fairness audit|$PY src/fairness_audit.py"
  "Five bootstrap-retrained models for the robustness check|$PY -m src.train_bootstrap_models --n 5"
  "Recourse evaluation incl. baselines, DiCE and robustness|$PY -m src.recourse_eval --n 300 --dice 20 --robustness"
)

if [[ "${1:-}" == "--list" ]]; then
  for i in "${!STEPS[@]}"; do printf "%2d  %s\n" "$((i + 1))" "${STEPS[$i]%%|*}"; done
  exit 0
fi

START=${1:-1}
if ! [[ "$START" =~ ^[0-9]+$ ]] || (( START < 1 || START > ${#STEPS[@]} )); then
  echo "start step must be 1-${#STEPS[@]}" >&2
  exit 2
fi

if (( START == 1 )) && [[ ! -f data/home_credit/application_train.csv ]]; then
  echo "data/home_credit/application_train.csv not found. Download the Kaggle data first (README, 'Data')." >&2
  exit 1
fi

mkdir -p data/processed models reports
for i in "${!STEPS[@]}"; do
  n=$((i + 1))
  (( n < START )) && continue
  desc=${STEPS[$i]%%|*}
  cmd=${STEPS[$i]#*|}
  echo
  echo "== Step $n/${#STEPS[@]}: $desc"
  echo "   $cmd"
  start_time=$SECONDS
  eval "$cmd"
  echo "   done in $((SECONDS - start_time)) s"
done
echo
echo "Pipeline complete. Run the tests with: $PY -m pytest tests backend/tests -q"
