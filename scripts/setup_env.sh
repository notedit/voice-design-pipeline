#!/usr/bin/env bash
# 从锁文件重建两个 venv。用法:bash scripts/setup_env.sh [main|enhance|all]
set -euo pipefail
cd "$(dirname "$0")/.."
what="${1:-all}"
mk() {  # name lockfile
  uv venv ".$1" --python 3.12 >/dev/null
  uv pip install --python ".$1/bin/python" -r "requirements/$2" >/dev/null
  echo "[$1] ready"
}
[[ "$what" == "main"    || "$what" == "all" ]] && mk venv main.lock.txt
[[ "$what" == "enhance" || "$what" == "all" ]] && mk venv-enhance enhance.lock.txt
echo "next: .venv/bin/python run.py doctor [project]"
