#!/usr/bin/env bash
# Full HeatCast-FM pipeline: data, labels, backtests, scores, figures, report.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python

$PY -m pytest -q
$PY scripts/01_download.py
$PY scripts/02_prepare.py
$PY scripts/03_backtest.py
$PY scripts/04_evaluate.py
$PY scripts/05_figures.py
$PY scripts/06_report_tables.py
(cd report && pdflatex -interaction=nonstopmode main.tex >/dev/null && bibtex main >/dev/null \
  && pdflatex -interaction=nonstopmode main.tex >/dev/null && pdflatex -interaction=nonstopmode main.tex >/dev/null)
cp report/main.pdf report/HeatCast-FM_Report.pdf
echo "Report: report/HeatCast-FM_Report.pdf"
