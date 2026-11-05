# 一键复现完整流水线（Windows PowerShell）
$ErrorActionPreference = "Stop"
python -m softmatter.cli synth   --out data/raw --n 2400
python -m softmatter.cli ingest  --root data/raw --out data/manifest.jsonl
python -m softmatter.cli clean   --manifest data/manifest.jsonl --out data/clean.jsonl
python -m softmatter.cli mixture --input data/clean.jsonl --out data/train.jsonl
python -m softmatter.cli cot     --input data/train.jsonl --out data/cot.jsonl
python -m softmatter.cli train   --input data/train.jsonl --out .
python -m softmatter.cli eval    --input data/train.jsonl --out outputs/eval_report.md
Write-Host "Done -> outputs/report.md, outputs/metrics.json"
