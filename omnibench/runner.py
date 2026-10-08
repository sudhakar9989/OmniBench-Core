import argparse
import datetime
import os
import subprocess
import sys
from pathlib import Path
from typing import List
from tabulate import tabulate

from vlmeval.config import supported_VLM
from vlmeval.dataset import build_dataset
from vlmeval.inference import infer_data_job
from vlmeval.inference_mt import infer_data_job_mt
from vlmeval.inference_video import infer_data_job_video
from vlmeval.smp import (
    build_eval_id, collect_run_benchmark_report, get_eval_file_format,
    get_logger, get_pred_file_format, get_pred_file_path, githash,
    load_env, setup_logger, timestr, upsert_dataset_status, upsert_run_status,
)

logger = get_logger("omnibench")

def get_available_gpu_indices() -> List[int]:
    cuda_env = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if cuda_env:
        return [int(x.strip()) for x in cuda_env.split(",") if x.strip().isdigit()]
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            capture_output=True, text=True, check=True
        )
        return [int(x.strip()) for x in r.stdout.strip().splitlines() if x.strip().isdigit()]
    except Exception:
        return []

def configure_gpu_cluster() -> None:
    world = int(os.environ.get("LOCAL_WORLD_SIZE", 1))
    rank = int(os.environ.get("LOCAL_RANK", 0))
    gpus = get_available_gpu_indices()
    if world > 1 and gpus and len(gpus) >= world:
        per = len(gpus) // world
        assigned = gpus[rank * per:(rank + 1) * per]
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, assigned))

class OmniBenchRunner:
    """Modular execution engine for multimodal benchmarking."""

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.run_id = build_eval_id()
        self.output_dir = Path(os.environ.get("MMEVAL_ROOT", args.work_dir))
        self.commit_hash = githash(digits=8)

    def prepare_dataset(self, dataset_name: str, model_name: str):
        kwargs = {}
        if dataset_name in ["MMLongBench_DOC", "DUDE", "DUDE_MINI", "SLIDEVQA"]:
            kwargs["model"] = model_name
        return build_dataset(dataset_name, **kwargs)

    def execute(self) -> None:
        log_dir = self.output_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        setup_logger(log_file=str(log_dir / f"omnibench_{self.run_id}_{timestr()}.log"))
        logger.info(f"Initialized OmniBench Core Session: {self.run_id} (Rev: {self.commit_hash})")

        for model_name in self.args.model:
            model_root = self.output_dir / model_name / self.run_id
            model_root.mkdir(parents=True, exist_ok=True)
            upsert_run_status(
                model_root, eval_id=self.run_id,
                created_at=datetime.datetime.now().astimezone().isoformat(),
                commit=self.commit_hash, argv=sys.argv, api_mode=False,
                world_size=int(os.environ.get("WORLD_SIZE", 1)),
                pred_format=get_pred_file_format(), eval_format=get_eval_file_format(),
                mode=self.args.mode, reuse=self.args.reuse,
                reuse_aux=self.args.reuse_aux, model_name=model_name,
            )
            model = supported_VLM[model_name]() if model_name in supported_VLM else model_name

            for dataset_name in self.args.data:
                logger.info(f"Dispatching Benchmark Evaluation: {dataset_name}")
                pred_file = get_pred_file_path(str(model_root), model_name, dataset_name, use_env_format=True)
                try:
                    dataset = self.prepare_dataset(dataset_name, model_name)
                    if not dataset:
                        continue
                    if self.args.mode != "eval":
                        if getattr(dataset, "MODALITY", None) == "VIDEO":
                            model = infer_data_job_video(
                                model, work_dir=model_root, model_name=model_name,
                                dataset=dataset, result_file=pred_file,
                                verbose=self.args.verbose, api_nproc=self.args.api_nproc)
                        elif getattr(dataset, "TYPE", None) == "MT":
                            model = infer_data_job_mt(
                                model, work_dir=model_root, model_name=model_name,
                                dataset=dataset, verbose=self.args.verbose,
                                api_nproc=self.args.api_nproc)
                        else:
                            model = infer_data_job(
                                model, work_dir=model_root, model_name=model_name,
                                dataset=dataset, verbose=self.args.verbose,
                                api_nproc=self.args.api_nproc)
                    if self.args.mode != "infer":
                        results = dataset.evaluate(pred_file)
                        if results is not None:
                            upsert_dataset_status(
                                run_dir=model_root, model_name=model_name,
                                dataset_name=dataset_name, status="done",
                                metrics_source=results, dataset_obj=dataset)
                except Exception as exc:
                    logger.exception(f"Exception during {model_name} on {dataset_name}: {exc}")
                    upsert_dataset_status(
                        run_dir=model_root, model_name=model_name,
                        dataset_name=dataset_name, status="error",
                        error_message=str(exc))
            self._log_summary(model_root)

    def _log_summary(self, target_dir: Path) -> None:
        rows = collect_run_benchmark_report(str(target_dir))
        if not rows:
            return
        records = []
        for row in rows:
            value = row.get("primary_metric_value", "-")
            records.append({
                "Benchmark": row.get("benchmark", "N/A"),
                "Status": row.get("status", "Completed"),
                "Primary Metric": row.get("primary_metric", "-"),
                "Score": f"{float(value):.4f}" if isinstance(value, (float, int)) else str(value),
                "Failed Samples": row.get("infer_failed", 0),
            })
        logger.info("Session Benchmark Matrix:\n" + tabulate(records, headers="keys"))

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="OmniBench-Core Command Center")
    p.add_argument("--data", nargs="+", required=True, help="Benchmark datasets")
    p.add_argument("--model", nargs="+", required=True, help="Target VLMs")
    p.add_argument("--work-dir", default="./outputs", help="Prediction output directory")
    p.add_argument("--mode", default="all", choices=["all", "infer", "eval"])
    p.add_argument("--api-nproc", type=int, default=16)
    p.add_argument("--reuse", action="store_true")
    p.add_argument("--reuse-aux", default="all", choices=["all", "infer", "none"])
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()

def main() -> None:
    load_env()
    configure_gpu_cluster()
    OmniBenchRunner(parse_args()).execute()

if __name__ == "__main__":
    main()
