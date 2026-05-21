#!/bin/sh
set -eu

export PYTHONPATH="/opt/pattern-learning-stack/src${PYTHONPATH:+:${PYTHONPATH}}"
restart_delay="${PATTERN_CLIENT_RESTART_DELAY_SECONDS:-5}"
stack_root="/opt/pattern-learning-stack"
archive_root="${stack_root}/data/dataset_archives"
dataset_root="${stack_root}/data/datasets"

if [ -d "$archive_root" ]; then
  find "$archive_root" -mindepth 2 -maxdepth 2 -type f -name '*.tar.gz' | sort | while read -r archive_path; do
    env_id="$(basename "$(dirname "$archive_path")")"
    dataset_name="$(basename "$archive_path" .tar.gz)"
    manifest_path="${dataset_root}/${env_id}/${dataset_name}/dataset_manifest.json"
    if [ -f "$manifest_path" ]; then
      continue
    fi
    mkdir -p "${dataset_root}/${env_id}"
    tar -xzf "$archive_path" -C "${dataset_root}/${env_id}"
  done
fi

while :; do
  set +e
  /opt/pattern-client-venv/bin/python -m pattern_client.main \
    --client-root /app \
    --execution-root "${stack_root}" \
    "$@"
  status=$?
  set -e

  if [ "$status" -eq 0 ]; then
    exit 0
  fi

  printf '%s\n' "pattern-client exited with status ${status}; restarting in ${restart_delay}s with a fresh registration." >&2
  sleep "$restart_delay"
done
