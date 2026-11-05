"""命令行入口: python -m softmatter.cli <command> [options]。"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

from .analyst import ReferenceAnalyst
from .config import load_config
from .data.clean import CleanConfig, clean_corpus
from .data.cot_builder import build_cot_dataset
from .data.mixture import (MixtureConfig, distribution_gap, mixture_statistics,
                           physical_distribution_check, stratified_mix)
from .data.readers import RawRecord, scan_corpus
from .data.schema import Sample, read_samples, write_jsonl
from .eval.evaluate import compare, evaluate_samples, write_report
from .inference.pipeline import analyze
from .physics.curves import SynthConfig, generate_dataset, write_raw_files
from .training.grpo_dapo import DAPOConfig, GaussianPolicy, train_dapo
from .training.reward import RewardConfig
from .training.sft import run_sft_pipeline


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _iter_manifest(path: Path) -> Iterable[RawRecord]:
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield RawRecord.from_dict(json.loads(line))


def prepare_data(cfg: Dict[str, Any], out_dir: Path,
                 n_samples: int = None) -> Dict[str, Any]:
    """合成 -> 原始文件 -> 流式 manifest -> 清洗 -> 配比 -> CoT。"""
    data_dir = out_dir / "data"
    raw_dir = data_dir / "raw"
    samples = generate_dataset(SynthConfig.from_dict(cfg["dataset"]), n_samples=n_samples)
    n_raw = write_raw_files(samples, raw_dir)
    write_jsonl(data_dir / "manifest.jsonl", (r.to_dict() for r in scan_corpus(raw_dir)))
    cleaned, clean_stats = clean_corpus(_iter_manifest(data_dir / "manifest.jsonl"),
                                        CleanConfig.from_dict(cfg["cleaning"]))
    write_jsonl(data_dir / "clean.jsonl", cleaned)
    _write_json(data_dir / "clean_stats.json", clean_stats)

    mixture_cfg = MixtureConfig.from_dict(cfg["mixture"])
    mixed = stratified_mix(cleaned, mixture_cfg)
    mixture_stats = mixture_statistics(mixed)
    mixture_stats["distribution_gap"] = distribution_gap(mixture_stats, mixture_cfg)
    mixture_stats["physics_check"] = physical_distribution_check(mixed, mixture_cfg)
    write_jsonl(data_dir / "train.jsonl", mixed)

    cot_records = build_cot_dataset(mixed, cfg.get("cot", {}))
    write_jsonl(data_dir / "cot.jsonl", cot_records)
    write_jsonl(data_dir / "train_with_cot.jsonl", mixed)
    _write_json(data_dir / "mixture_stats.json", mixture_stats)
    return {"samples": mixed, "n_raw_files": n_raw, "clean_stats": clean_stats,
            "mixture_stats": mixture_stats, "cot_records": cot_records}


def run_training(cfg: Dict[str, Any], samples: List[Sample],
                 out_dir: Path) -> Dict[str, Any]:
    """三阶段 SFT -> DAPO/GRPO -> 评估与报告。"""
    outputs = out_dir / "outputs"
    eval_size = int(cfg["sft"].get("eval_size", 300))
    eval_samples = samples[:min(eval_size, len(samples) // 3)]
    train_samples = samples[len(eval_samples):]
    if not train_samples:
        train_samples = samples
    baseline_analyst = ReferenceAnalyst(stage=3)
    _, baseline = evaluate_samples(eval_samples, baseline_analyst, cfg.get("eval", {}))

    analyst = ReferenceAnalyst(stage=1)
    stages = run_sft_pipeline(train_samples, analyst, cfg["sft"], eval_samples,
                              cfg.get("eval", {}))
    dapo = train_dapo(train_samples, analyst, DAPOConfig.from_dict(cfg["dapo"]),
                      RewardConfig.from_dict(cfg["reward"]),
                      holdout=eval_samples[:min(120, len(eval_samples))],
                      eval_cfg=cfg.get("eval", {}))
    _, final = evaluate_samples(eval_samples, analyst, cfg.get("eval", {}),
                                policy=dapo["policy"])
    deltas = compare(baseline, final)

    analyst.save(outputs / "analyst.json")
    _write_json(outputs / "policy.json", dapo["policy"].to_dict())
    _write_json(outputs / "sft_stages.json", stages)
    _write_json(outputs / "dapo_history.json", dapo["history"])
    metrics = {"baseline": baseline, "final": final, "deltas": deltas,
               "sft_stages": stages, "dapo": {"n_hardcases": dapo["n_hardcases"],
                                              "final": dapo["final"]}}
    _write_json(outputs / "metrics.json", metrics)
    return {"metrics": metrics, "dapo_history": dapo["history"], "analyst": analyst,
            "baseline": baseline, "final": final}


def run_demo(cfg: Dict[str, Any], out_dir: Path, n_samples: int = None) -> Dict[str, Any]:
    prepared = prepare_data(cfg, out_dir, n_samples)
    trained = run_training(cfg, prepared["samples"], out_dir)
    write_report(out_dir / "outputs" / "report.md", trained["final"],
                 stages=trained["metrics"]["sft_stages"],
                 dapo_history=trained["dapo_history"],
                 extra={
                     "raw_files": prepared["n_raw_files"],
                     "clean_acceptance_rate": round(prepared["clean_stats"]["acceptance_rate"], 4),
                     "porod_law_compliance": round(
                         prepared["mixture_stats"]["physics_check"]["porod_law_compliance"], 4),
                     "regime_gap": round(
                         prepared["mixture_stats"]["distribution_gap"]["max_regime_gap"], 4),
                 })
    baseline, final = trained["baseline"], trained["final"]
    print("\n=== SoftMatter-LLM demo ===")
    print("原始文件 %d -> 清洗接受率 %.3f -> 配比样本 %d" %
          (prepared["n_raw_files"], prepared["clean_stats"]["acceptance_rate"],
           len(prepared["samples"])))
    print("Porod 合规率 %.3f | 配比最大偏差 %.3f" %
          (prepared["mixture_stats"]["physics_check"]["porod_law_compliance"],
           prepared["mixture_stats"]["distribution_gap"]["max_regime_gap"]))
    print("SFT 阶段相判定: " + " -> ".join("%.3f" % s["phase_accuracy"]
                                       for s in trained["metrics"]["sft_stages"]))
    print("相判定准确率   %.3f -> %.3f (%+.3f)" %
          (baseline["phase_accuracy"], final["phase_accuracy"],
           trained["metrics"]["deltas"]["phase_accuracy_delta"]))
    print("参数提取准确率 %.3f -> %.3f (%+.3f)" %
          (baseline["parameter_accuracy"], final["parameter_accuracy"],
           trained["metrics"]["deltas"]["parameter_accuracy_delta"]))
    print("相图综合分     %.3f" % final["phase_diagram"]["score"])
    print("报告: %s" % (out_dir / "outputs" / "report.md"))
    return trained["metrics"]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="softmatter", description="软物质散射/流变智能解析系统")
    parser.add_argument("--config", default=None, help="覆盖默认配置的 JSON 路径")
    sub = parser.add_subparsers(dest="command", required=True)

    synth = sub.add_parser("synth", help="合成散射/流变数据与原始 .dat 文件")
    synth.add_argument("--out", default="data/raw")
    synth.add_argument("--n", type=int, default=None)

    ingest = sub.add_parser("ingest", help="流式扫描 10 万级实验文件并写 manifest")
    ingest.add_argument("--root", default="data/raw")
    ingest.add_argument("--out", default="data/manifest.jsonl")

    clean = sub.add_parser("clean", help="清洗 + Porod 定律区校验")
    clean.add_argument("--manifest", default="data/manifest.jsonl")
    clean.add_argument("--out", default="data/clean.jsonl")

    mixture = sub.add_parser("mixture", help="多维数据配比")
    mixture.add_argument("--input", default="data/clean.jsonl")
    mixture.add_argument("--out", default="data/train.jsonl")

    cot = sub.add_parser("cot", help="构造 CoT 训练样本")
    cot.add_argument("--input", default="data/train.jsonl")
    cot.add_argument("--out", default="data/cot.jsonl")

    train = sub.add_parser("train", help="三阶段 SFT + DAPO/GRPO")
    train.add_argument("--input", default="data/train.jsonl")
    train.add_argument("--out", default=".")

    evaluate = sub.add_parser("eval", help="评估已保存的分析器")
    evaluate.add_argument("--model", default="outputs/analyst.json")
    evaluate.add_argument("--policy", default="outputs/policy.json")
    evaluate.add_argument("--input", default="data/train.jsonl")
    evaluate.add_argument("--out", default="outputs/eval_report.md")

    infer = sub.add_parser("infer", help="单条谱图端到端推理")
    infer.add_argument("--model", default="outputs/analyst.json")
    infer.add_argument("--input", default="data/train.jsonl")
    infer.add_argument("--id", default=None)

    demo = sub.add_parser("demo", help="一键端到端复现")
    demo.add_argument("--out", default=".")
    demo.add_argument("--n", type=int, default=None)
    return parser


def main(argv: List[str] = None) -> int:
    args = _build_parser().parse_args(argv)
    cfg = load_config(args.config)
    if args.command == "synth":
        samples = generate_dataset(SynthConfig.from_dict(cfg["dataset"]), n_samples=args.n)
        write_jsonl(Path(args.out).parent / "raw.jsonl", samples)
        print("wrote %d raw files to %s" % (write_raw_files(samples, args.out), args.out))
    elif args.command == "ingest":
        records = (r.to_dict() for r in scan_corpus(args.root))
        print("manifest rows: %d" % write_jsonl(args.out, records))
    elif args.command == "clean":
        samples, stats = clean_corpus(_iter_manifest(Path(args.manifest)),
                                      CleanConfig.from_dict(cfg["cleaning"]))
        write_jsonl(args.out, samples)
        print(json.dumps(stats, ensure_ascii=False))
    elif args.command == "mixture":
        mixture_cfg = MixtureConfig.from_dict(cfg["mixture"])
        mixed = stratified_mix(read_samples(args.input), mixture_cfg)
        write_jsonl(args.out, mixed)
        print(json.dumps(mixture_statistics(mixed), ensure_ascii=False))
    elif args.command == "cot":
        records = build_cot_dataset(read_samples(args.input), cfg.get("cot", {}))
        write_jsonl(args.out, records)
        print("CoT records: %d" % len(records))
    elif args.command == "train":
        run_training(cfg, read_samples(args.input), Path(args.out))
        print("training done -> %s" % (Path(args.out) / "outputs" / "metrics.json"))
    elif args.command == "eval":
        analyst = ReferenceAnalyst.load(args.model)
        policy = None
        if Path(args.policy).exists():
            policy = GaussianPolicy.from_dict(
                json.loads(Path(args.policy).read_text(encoding="utf-8")))
        _, summary = evaluate_samples(read_samples(args.input), analyst,
                                      cfg.get("eval", {}), policy=policy)
        write_report(args.out, summary)
        print(json.dumps({k: summary[k] for k in ("phase_accuracy", "parameter_accuracy")},
                         ensure_ascii=False))
    elif args.command == "infer":
        samples = read_samples(args.input)
        sample = next((s for s in samples if s.id == args.id), samples[0])
        result = analyze(sample, ReferenceAnalyst.load(args.model))
        print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
    elif args.command == "demo":
        run_demo(cfg, Path(args.out), args.n)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
