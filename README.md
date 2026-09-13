# 🚀 OmniBench-Core

OmniBench-Core is an automated, modular evaluation framework for high-throughput benchmarking of Vision-Language Models (VLMs), Vision Agents, and Multimodal LLMs across image, video, document, and spatial tasks.

## Features
- Unified multimodal inference/evaluation pipeline
- Distributed GPU execution support
- Checkpoint/status tracking
- Extensible dataset and scoring architecture
- CLI command: `omnibench`

## Install
```bash
pip install -e .
```

## Examples
```bash
omnibench --model gpt-4o --data MMBench_DEV_EN --mode all
omnibench --model Qwen2-VL-7B-Instruct --data MathVista MMBench_DEV_EN --mode infer
```

## Important compatibility note
The supplied source specification rebrands the runner to `omnibench`, but `runner.py` still imports the legacy `vlmeval.*` modules. Those underlying modules are therefore still required unless the imports are migrated.

## License
Apache 2.0
