# Physics Pattern Learning

Source code for the Physics Pattern Learning project.

This repository contains the training, evaluation, server, annotation, summary,
and analysis code used in the project, along with the reusable dependency repos
tracked as submodules under `deps/`.

## Repository Structure

- `src/pattern_learning`: pattern learning and evaluation code
- `src/ensemble_library_training`: ensemble and canonical library training
- `src/video_annotations`: video rendering and annotation
- `src/summaries`: summary generation and profiling
- `src/code_convergence`: consistency and convergence analysis
- `src/statistics`: aggregate analysis utilities
- `src/reward_programs`: reward-program experiments
- `src/human_survey`: survey app and analysis scripts
- `src/pattern_server`: server and dashboard code
- `src/pattern_client`: client runtime
- `deps/`: reusable dependency repositories tracked as submodules

## Setup

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH="$PWD/src"
```

## Running

Start the server:

```bash
python -m pattern_server.main
```

Build the client image:

```bash
podman build -f docker/client.Dockerfile -t pattern-learning-client .
```
