#!/usr/bin/env bash
# 对 docs/detail.html 做 headless 截图（供 Cloud Agent PR 验证）。
# Chrome 在部分环境下会在「bytes written」之后仍不退出，故必须外层 timeout + 以 PNG 是否落盘为成功判据。
#
# 用法（在仓库根目录）:
#   ./scripts/screenshot_site_detail.sh wiki-methods-sonic-motion-tracking
#   ./scripts/screenshot_site_detail.sh wiki-methods-sonic-motion-tracking /path/out.png
#   ./scripts/screenshot_site_detail.sh wiki-concepts-sim2real /path/out.png detail-sources
#     第三个参数为页面锚点（不带 #），例如 detail-sources 可定位到「来源链接」区块再截图视口。
#
# 环境变量:
#   PORT   默认 8765
#   CHROME 默认 google-chrome
#   SCREENSHOT_HEIGHT  当提供锚点时视口高度，默认 1000（宽度固定 1280）
#   带锚点时改用 Node + puppeteer-core 控制已安装的 Chrome（见 screenshot_detail_anchor_viewport.cjs），
#   不再使用 headless --virtual-time-budget，以便在 Mermaid / fetch 完成后再滚动到目标区块。

set -u

PAGE_ID="${1:?第一个参数: detail 页 id，如 wiki-methods-sonic-motion-tracking}"
OUT_PATH="${2:-}"
ANCHOR_RAW="${3:-}"
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
ANCHOR="${ANCHOR_RAW#\#}"
WIN_H=2400

if [[ -n "$ANCHOR" ]]; then
  WIN_H="${SCREENSHOT_HEIGHT:-1000}"
fi

if [[ -n "$ANCHOR" ]]; then
  set +e
  timeout 120 env PUPPETEER_EXECUTABLE_PATH="$(command -v "$CHROME" 2>/dev/null || echo "$CHROME")" \
    node "$ROOT/scripts/screenshot_detail_anchor_viewport.cjs" "$URL" "$OUT_PATH" "$WIN_H" "$ANCHOR"
  NODE_RC=$?
  set -e
  if [[ "$NODE_RC" -eq 124 ]]; then
    echo "screenshot_site_detail: 警告 Node/Puppeteer 由 timeout 结束，退出码 124" >&2
  elif [[ "$NODE_RC" -ne 0 ]]; then
    echo "screenshot_site_detail: Node/Puppeteer 退出码=$NODE_RC" >&2
    exit "$NODE_RC"
  fi
else
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
    --window-size=1280,"$WIN_H" \
    --virtual-time-budget=20000 \
    --disable-remote-fonts \
    --screenshot="$OUT_PATH" \
    "$URL"
  CHROME_RC=$?
  set -e

  if [[ "$CHROME_RC" -eq 124 ]]; then
    echo "screenshot_site_detail: 已生成 $OUT_PATH（Chrome 由 timeout 结束，退出码 124 为预期内）"
  elif [[ "$CHROME_RC" -ne 0 ]]; then
    echo "screenshot_site_detail: 警告 Chrome 退出码=$CHROME_RC，但 PNG 已生成: $OUT_PATH" >&2
  fi
fi

BYTES=0
if [[ -f "$OUT_PATH" ]]; then
  BYTES=$(wc -c <"$OUT_PATH" | tr -d ' ')
fi

if [[ "$BYTES" -lt 10000 ]]; then
  echo "screenshot_site_detail: 失败，输出过小或不存在: $OUT_PATH ($BYTES bytes)" >&2
  exit 1
fi

echo "$OUT_PATH"
