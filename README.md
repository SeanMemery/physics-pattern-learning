# Discovering High Level Patterns from Simulation Traces

This repository contains the code used for the paper **"Discovering High
Level Patterns from Simulation Traces"** by Sean Memery and Kartic Subr
(University of Edinburgh).

The project learns interpretable pattern detectors from physics simulation
traces, converts traces into sparse annotated representations, and uses those
annotations for downstream tasks including summarization, reasoning, and reward
program synthesis.

## Paper Overview

The core idea of the paper is to translate low-level simulation traces into
high-level structural patterns that are easier for language models to use. The
repository includes code for:

- learning pattern detectors from simulation traces via evolutionary program
  synthesis
- building and applying pattern libraries and ensemble libraries
- generating annotated traces for downstream tasks
- summarization, analysis, and reward-program experiments
- server infrastructure and a client container for distributed evaluation

## Repository Layout

- `src/pattern_learning`: pattern learning, detector application, and evaluation
- `src/ensemble_library_training`: ensemble and canonical library training
- `src/video_annotations`: trace visualization and annotation rendering
- `src/summaries`: summary generation and profiling
- `src/code_convergence`: consistency and convergence analysis
- `src/statistics`: aggregate statistics and reporting
- `src/reward_programs`: reward-program experiments
- `src/human_survey`: human survey application and analysis
- `src/pattern_server`: server and dashboard code
- `src/pattern_client`: client runtime
- `deps/`: git submodules for shared libraries used by this repository

## Setup

```bash
git submodule update --init --recursive
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH="$PWD/src"
```

## Main Entry Points

Start the server:

```bash
python -m pattern_server.main
```

Build the client image:

```bash
podman build -f docker/client.Dockerfile -t pattern-learning-client .
```

## Citation

If you use this code, please cite the paper:

```bibtex
@misc{memery2026discoveringhighlevelpatterns,
  title={Discovering High Level Patterns from Simulation Traces}, 
  author={Sean Memery and Kartic Subr},
  year={2026},
  eprint={2602.10009},
  archivePrefix={arXiv},
  primaryClass={cs.AI},
  url={https://arxiv.org/abs/2602.10009}, 
}
```
