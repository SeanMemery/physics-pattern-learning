# Human Survey Migration Plan

## Goal

Build a public-facing Prolific study from the current `/summaries` outputs in `pattern-learning-server`, with extracted media/summaries stored under `human_survey/data`, and served publicly through Tailscale Funnel.

## Current source of truth

The source content lives in `pattern-learning-server/src/pattern_server/task_summaries.py` and is surfaced in the `/summaries` page.

Each summary entry currently has:

- `env_id`
- `source_run_id`
- `task_id`
- `video_path`
- `overhead_video_path` for PoolTool
- `initial_image_path`
- `summary_payload.variants.*.summary_text`

Current summary variants present in `/summaries`:

- `image_only`
- `supervised`
- `unsupervised`

## Current inventory snapshot

Snapshot taken on 2026-05-06 from `list_completed_task_summaries(...)`.

### PHYRE

- entries: `18`
- with playable video: `14`
- with initial image: `18`
- `image_only`: `14`
- `supervised`: `18`
- `unsupervised`: `18`

### IPHYRE

- entries: `14`
- with playable video: `12`
- with initial image: `14`
- `image_only`: `12`
- `supervised`: `14`
- `unsupervised`: `14`

### KINETIX

- entries: `17`
- with playable video: `5`
- with initial image: `17`
- `image_only`: `4`
- `supervised`: `15`
- `unsupervised`: `17`

### POOLTOOL

- entries: `15`
- with playable video: `15`
- with overhead video: `15`
- with initial image: `15`
- `image_only`: `10`
- `supervised`: `15`
- `unsupervised`: `15`

## Gaps to record during extraction

For each extracted task, record:

- whether `video` exists
- whether `overhead_video` exists
- whether `initial_image` exists
- whether `image_only` summary exists
- whether `supervised` summary exists
- whether `unsupervised` summary exists

This becomes the completion ledger for the export pass.

## Proposed export layout

Store all survey-ready assets under `human_survey/data`.

```text
human_survey/data/
  manifest.json
  extraction_report.md
  items/
    phyre/
      <task_slug>/
        item.json
        video.mp4
        initial_image.png
        summary_image_only.json
        summary_supervised.json
        summary_unsupervised.json
    iphyre/
      <task_slug>/
        ...
    kinetix/
      <task_slug>/
        ...
    pooltool/
      <task_slug>/
        video.mp4
        overhead_video.mp4
        initial_image.png
        ...
```

`item.json` should contain:

- `env_id`
- `task_id`
- `source_run_id`
- `item_id`
- `video_path`
- `overhead_video_path`
- `initial_image_path`
- `available_summary_types`
- `annotation_pattern_counts`
- `summary_sources`

`manifest.json` should contain:

- study metadata
- per-env item lists
- global flat item list
- counts by env and variant
- selection-ready fields so the frontend does not need to crawl directories

`extraction_report.md` should list:

- totals by env
- missing assets by env
- skipped tasks and reason
- whether each env is ready for study sampling

## Study design changes from the current implementation

The current study expects:

- two library buckets: `supervised_library`, `unified_library`
- three summary modes:
  - `summary_video_with_annotations`
  - `summary_video_only`
  - `summary_annotations_only`

That no longer matches the current `/summaries` product.

### New study shape

Use the current summary variants directly:

- `image_only`
- `supervised`
- `unsupervised`

Use one unified item pool across environments, with environment metadata retained.

For each trial, show:

- the video
- one sampled summary variant
- a rating question

Optional design extension:

- stratify sampling by env so Prolific participants see a balanced env mix
- avoid PoolTool side-view by always using `overhead_video.mp4` when available

## Frontend changes required

### `experiment.js`

Replace old assumptions:

- remove `supervised_library` vs `unified_library`
- remove old summary filenames
- load the new `data/manifest.json`
- sample from exported `items`
- support variant availability per item instead of assuming every item has every summary
- record:
  - `env_id`
  - `item_id`
  - `summary_variant`
  - `source_run_id`
  - `task_id`

Add support for Prolific query params:

- `PROLIFIC_PID`
- `STUDY_ID`
- `SESSION_ID`

Persist them with every saved response.

### `index.html`

Keep the jsPsych shell, but update the title/instructions to match the new study.

### `style.css`

Likely minor changes only:

- ensure large videos fit without scroll
- improve mobile handling if served publicly

### `save_data.php`

Current PHP append-only file storage is acceptable for a first pass, but should be updated to:

- write into `human_survey/data/responses/`
- sanitize incoming IDs
- add coarse server-side metadata:
  - remote IP if desired
  - user agent
  - received timestamp
- return clear error responses

## Extraction implementation plan

1. Write an export script in `human_survey/` that imports `list_completed_task_summaries`.
2. Filter to entries with a playable video.
3. Prefer PoolTool overhead video over side/askew view.
4. Copy media into `human_survey/data/items/<env>/<task_slug>/`.
5. Write one JSON file per available summary variant.
6. Build `manifest.json`.
7. Build `extraction_report.md`.

## Selection rules for the first export

Use these rules unless changed:

1. Include only tasks with a playable video.
2. Include any available summary variants for that task.
3. Do not require all three summary variants to keep an item.
4. Mark missing variants explicitly in the manifest.
5. For PoolTool, prefer top-down video.

## Public deployment plan with Tailscale Funnel

Serve the study locally, then expose it via Funnel.

Recommended local serving model:

- run a small local web server on a single port
- serve static assets and response collection from the same app
- avoid relying on PHP if the public deployment host does not already run it

Recommended implementation:

- replace `save_data.php` with a small Python server, or add a minimal FastAPI app in this folder
- serve:
  - `/`
  - `/data/...`
  - `/api/save`

Then expose that local server through Tailscale Funnel.

Notes:

- Funnel is public internet exposure over HTTPS.
- The current CLI uses `tailscale funnel`.
- Allowed HTTPS listen ports are `443`, `8443`, or `10000`.
- Funnel proxies to a local service such as `localhost:3000`.
- Serve and Funnel cannot use the same port number simultaneously.

## Recommended Funnel runbook

Example target flow:

1. Run the survey app locally on `127.0.0.1:3000`.
2. Expose it:
   `tailscale funnel --bg localhost:3000`
3. Confirm status:
   `tailscale funnel status`
4. Use the issued `https://<node>.<tailnet>.ts.net` URL in Prolific.

## Risks

- Kinetix currently has weak video coverage, so it may need to be down-weighted or excluded from the first participant wave.
- The current PHP save path is fine for local use but brittle for public deployment.
- Media volume may make initial preload too heavy if the study samples many tasks at once.

## Next execution pass

1. Implement the export script.
2. Generate `human_survey/data/manifest.json`.
3. Generate `human_survey/data/extraction_report.md`.
4. Refactor the survey app to use the new manifest and Prolific params.
5. Replace PHP persistence with a single-process Python app.
6. Smoke test locally.
7. Expose with Tailscale Funnel.
