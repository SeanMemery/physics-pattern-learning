## pooltool-RL

Standalone PoolTool reward-program experiments with no web UI.

### Experiments

- `single-scene`: sample one scene, create a DSL reward, optimize an action, and render the result.
- `verifier-opt`: optimize simple verifier-backed rewards across scenes, seeds, and budgets.
- `verifier-train`: train two value models per verifier-backed goal and compare reward-trained vs verifier-trained behavior.
- `unverified-opt`: optimize richer unverified rewards across scenes, seeds, and budgets.

### Commands

```bash
python -m reward_programs.pooltool_rl.cli single-scene sample
python -m reward_programs.pooltool_rl.cli single-scene annotate-video --run-dir single_scene/runs/<run_id>
python -m reward_programs.pooltool_rl.cli annotations render --trace-path <trace.json> --video-path <simulation_overhead.mp4>
python -m reward_programs.pooltool_rl.cli verifier-opt run
python -m reward_programs.pooltool_rl.cli verifier-train run
python -m reward_programs.pooltool_rl.cli unverified-opt run
```

### Layout

- `configs/config.yml`: LLM and experiment defaults
- `configs/goals.example.json`: example goal list
- `configs/goals.verified.json`: simple verifier-backed reward collection
- `configs/goals.unverified.json`: richer unverified reward collection
- `data/scenes/test_pooltool_100/`: fixed 100-scene PoolTool test dataset
- `data/libraries/pooltool/supervised/`: copied supervised ensemble library
- `single_scene/`: single-scene workflow outputs and docs
- `experiments/verifier_opt/`: verifier optimization docs
- `experiments/verifier_train/`: verifier training docs
- `experiments/unverified_opt/`: unverified optimization docs
- `artifacts/`: one timestamped run directory per CLI invocation

### Annotation Overlays

The repo can render supervised-library annotation overlays directly onto PoolTool overhead videos.

Use either:

```bash
python -m reward_programs.pooltool_rl.cli single-scene annotate-video --run-dir single_scene/runs/<run_id>
```

or the generic trace/video path form:

```bash
python -m reward_programs.pooltool_rl.cli annotations render \
  --trace-path path/to/trace.json \
  --video-path path/to/simulation_overhead.mp4 \
  --annotation-file path/to/trace.evidence.json
```

This writes:
- an annotated mp4
- a `.overlay.json` sidecar with the flattened overlay events used for rendering

### Notes

- The current repo is fully standalone and does not import from the old pattern-learning repositories.
- The DSL now supports both direct PoolTool predicates and `PATTERN(...)` queries over the local supervised pattern library.
- Verified experiments use exact handwritten binary verifiers.
- Unverified experiments optimize DSL reward only and inspect the resulting artifacts.
