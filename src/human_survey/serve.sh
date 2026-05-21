#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$script_dir"

# Default to 0.0.0.0 so the UI is reachable from other machines on the network.
# Override with HOST=127.0.0.1 if you want it local-only.
host="${HOST:-0.0.0.0}"
port="${1:-3000}"

echo "Serving human survey from $script_dir"
echo "Visit: http://localhost:$port (bound to $host)"
echo "Public with Funnel: tailscale funnel --bg localhost:$port"

exec python3 server.py --host "$host" --port "$port"
