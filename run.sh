#!/bin/sh
set -eu
cd "$(dirname "$0")"
if [ ! -x .venv/bin/python ]; then
  echo 'Missing local environment. Follow README.md setup instructions.' >&2
  exit 1
fi
if [ ! -f artifacts/scene/fitted.npz ] || [ ! -d node_modules/three ]; then
  echo 'Missing trained assets or viewer dependency. Follow README.md reproduction instructions.' >&2
  exit 1
fi
echo 'SceneQuery: http://127.0.0.1:8765 — press Ctrl+C to stop.'
exec .venv/bin/python -m uvicorn scenequery.server:app --host 127.0.0.1 --port 8765
