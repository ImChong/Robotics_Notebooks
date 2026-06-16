#!/usr/bin/env bash
# 对 docs/detail.html 做 headless 截图（供 Cloud Agent PR 验证）。
# 统一使用 Node + puppeteer-core 等待元信息异步渲染完成后再截图。
#
# 用法（在仓库根目录）:
#   ./scripts/screenshot_site_detail.sh wiki-methods-sonic-motion-tracking
#   ./scripts/screenshot_site_detail.sh wiki-methods-sonic-motion-tracking /path/out.png
#   ./scripts/screenshot_site_detail.sh wiki-concepts-sim2real /path/to/out.png detail-sources
#   ./scripts/screenshot_site_detail.sh wiki-concepts-vision-backbones /path/to/out.png detailMetaPanel
#     第三个参数为页面锚点（不带 #），例如 detail-sources / detailMetaPanel。
#
# 环境变量:
#   PORT   默认 8765
#   CHROME 默认 google-chrome
#   SCREENSHOT_HEIGHT  视口高度，默认无锚点 2400、有锚点 1000

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
if [[ -n "$ANCHOR" ]]; then
  WIN_H="${SCREENSHOT_HEIGHT:-1000}"
else
  WIN_H="${SCREENSHOT_HEIGHT:-2400}"
fi

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

BYTES=0
if [[ -f "$OUT_PATH" ]]; then
  BYTES=$(wc -c <"$OUT_PATH" | tr -d ' ')
fi

if [[ "$BYTES" -lt 10000 ]]; then
  echo "screenshot_site_detail: 失败，输出过小或不存在: $OUT_PATH ($BYTES bytes)" >&2
  exit 1
fi

echo "$OUT_PATH"
