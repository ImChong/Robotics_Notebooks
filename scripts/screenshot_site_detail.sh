#!/usr/bin/env bash
# 对 docs/detail.html 做 headless 截图（供 Cloud Agent PR 验证）。
# Chrome 在部分环境下会在「bytes written」之后仍不退出，故必须外层 timeout + 以 PNG 是否落盘为成功判据。
#
# 用法（在仓库根目录）:
#   ./scripts/screenshot_site_detail.sh wiki-methods-sonic-motion-tracking
#   ./scripts/screenshot_site_detail.sh wiki-methods-sonic-motion-tracking /path/out.png
#
# 环境变量:
#   PORT   默认 8765
#   CHROME 默认 google-chrome

set -u

PAGE_ID="${1:?第一个参数: detail 页 id，如 wiki-methods-sonic-motion-tracking}"
OUT_PATH="${2:-}"
PORT="${PORT:-8765}"
CHROME="${CHROME:-google-chrome}"

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
if [[ -z "$OUT_PATH" ]]; then
  mkdir -p "$ROOT/.cursor-artifacts/screenshots"
  OUT_PATH="$ROOT/.cursor-artifacts/screenshots/${PAGE_ID}.png"
else
  mkdir -p "$(dirname "$OUT_PATH")"
fi

HTTP_LOG="${TMPDIR:-/tmp}/http-screenshot-${PORT}.log"

cleanup() {
  if [[ -n "${HTTP_PID:-}" ]] && kill -0 "$HTTP_PID" 2>/dev/null; then
    kill "$HTTP_PID" 2>/dev/null || true
    wait "$HTTP_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT

cd "$ROOT/docs"
python3 -m http.server "$PORT" >"$HTTP_LOG" 2>&1 &
HTTP_PID=$!
sleep 2

URL="http://127.0.0.1:${PORT}/detail.html?id=${PAGE_ID}"
# 75s：覆盖 virtual-time-budget + 慢磁盘；Chrome 不退出时由 timeout 强杀
DEBUG_PORT=$((20000 + RANDOM % 10000))
# 避免与残留 headless 实例争用默认调试端口（常见 9222）
set +e
timeout 75 "$CHROME" \
  --headless=new \
  --disable-gpu \
  --no-sandbox \
  --disable-dev-shm-usage \
  --remote-debugging-port="$DEBUG_PORT" \
  --user-data-dir="/tmp/chrome-headless-screenshot-$$" \
  --window-size=1280,2400 \
  --virtual-time-budget=20000 \
  --disable-remote-fonts \
  --screenshot="$OUT_PATH" \
  "$URL"
CHROME_RC=$?
set -e

BYTES=0
if [[ -f "$OUT_PATH" ]]; then
  BYTES=$(wc -c <"$OUT_PATH" | tr -d ' ')
fi

if [[ "$BYTES" -lt 10000 ]]; then
  echo "screenshot_site_detail: 失败，输出过小或不存在: $OUT_PATH ($BYTES bytes)，Chrome 退出码=$CHROME_RC" >&2
  exit 1
fi

if [[ "$CHROME_RC" -eq 124 ]]; then
  echo "screenshot_site_detail: 已生成 $OUT_PATH（Chrome 由 timeout 结束，退出码 124 为预期内）"
elif [[ "$CHROME_RC" -ne 0 ]]; then
  echo "screenshot_site_detail: 警告 Chrome 退出码=$CHROME_RC，但 PNG 已生成: $OUT_PATH" >&2
fi

echo "$OUT_PATH"
