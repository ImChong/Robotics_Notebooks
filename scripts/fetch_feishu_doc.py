#!/usr/bin/env python3
"""fetch_feishu_doc.py — 抓取飞书公开文档（*.feishu.cn/wiki|docx）正文纯文本。

背景：WebFetch 访问飞书文档会被 302 到登录页；普通浏览器访客视图虽可打开「互联网公开」文档，
但文档是 JS 渲染 + 虚拟滚动，只有当前可见的块在 DOM 里，一次性读取只能拿到目录或开头几段。
本脚本用 Chromium（Playwright）打开页面，找到真正的滚动容器，边滚边按 data-block-id 累积
文本，滚到底后输出（2026-10-12 在 Claude Code 云容器实测：70 个块、约 4400 字）。

用法：
    python3 scripts/fetch_feishu_doc.py <url> [-o out.txt] [--links]

`--links` 在文末附「正文超链接」列表（文字 → 地址），用于取论文/原帖等链接。

依赖：`pip install playwright`，并已有 Chromium（云容器预装在 /opt/pw-browsers；
也可用环境变量 CHROMIUM_PATH 指定，或 `playwright install chromium`）。

边界：只适用于「互联网公开」文档；需要登录或无权限的页面会停在登录页，脚本以退出码 2 报错。
抓取内容多为转载，使用时保留原文链接并遵守原作者版权声明。
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

DOC_SELECTOR = ".page-block-children"

_JS_SCROLL = """(y) => {
  let el = document.querySelector('.page-block-children');
  while (el) {
    const oy = getComputedStyle(el).overflowY;
    if (el.scrollHeight > el.clientHeight + 50 && (oy === 'auto' || oy === 'scroll' || oy === 'overlay')) {
      el.scrollTop = y;
      return [el.scrollTop, el.scrollHeight, el.clientHeight];
    }
    el = el.parentElement;
  }
  window.scrollTo(0, y);
  return [-1, 0, 0];
}"""

_JS_BLOCKS = """() => Array.from(document.querySelectorAll('.page-block-children [data-block-id]')).map(
  e => [e.getAttribute('data-block-id'), e.innerText || '',
        Array.from(e.querySelectorAll('a[href]')).map(a => [a.innerText.trim(), a.href])])"""


def find_chromium() -> str | None:
    env = os.environ.get("CHROMIUM_PATH")
    if env:
        return env
    root = os.environ.get("PLAYWRIGHT_BROWSERS_PATH", "/opt/pw-browsers")
    for pattern in ("chromium-*/chrome-linux*/chrome", "chromium/chrome-linux*/chrome"):
        hits = sorted(glob.glob(os.path.join(root, pattern)))
        if hits:
            return hits[-1]
    return None


def leaf_blocks(texts: list[str]) -> list[str]:
    """去掉被其他块完整包含的容器块，只留叶子块，保持出现顺序。"""
    kept = []
    for i, t in enumerate(texts):
        if t.strip() and not any(
            j != i and t in u and len(u) > len(t) for j, u in enumerate(texts)
        ):
            kept.append(t.strip())
    return kept


def fetch(url: str, max_steps: int = 150) -> tuple[list[str], list[tuple[str, str]]]:
    from playwright.sync_api import sync_playwright

    seen: dict[str, str] = {}
    order: list[str] = []
    links: dict[str, str] = {}
    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path=find_chromium(), args=["--no-sandbox"])
        try:
            ctx = browser.new_context(
                user_agent=(
                    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
                ),
                viewport={"width": 1400, "height": 1000},
                locale="zh-CN",
            )
            page = ctx.new_page()
            page.goto(url, timeout=60000, wait_until="networkidle")
            if "login" in page.url:
                raise PermissionError("被重定向到登录页：文档不是「互联网公开」或无访客权限")
            page.wait_for_selector(DOC_SELECTOR, timeout=30000)
            page.wait_for_timeout(8000)  # 首屏块异步渲染
            y, last, stuck = 0, -2, 0
            for _ in range(max_steps):
                top = page.evaluate(_JS_SCROLL, y)
                page.wait_for_timeout(500)
                for bid, txt, hrefs in page.evaluate(_JS_BLOCKS):
                    if bid not in seen:
                        order.append(bid)
                    if len(txt) > len(seen.get(bid, "")):
                        seen[bid] = txt
                    for label, href in hrefs:
                        links.setdefault(href, label)
                stuck = stuck + 1 if top[0] == last else 0
                last = top[0]
                if stuck >= 3:
                    break
                y += 700
        finally:
            browser.close()
    return leaf_blocks([seen[b] for b in order]), [(v, k) for k, v in links.items()]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("url", help="https://<tenant>.feishu.cn/wiki/<token> 或 /docx/<token>")
    ap.add_argument("-o", "--output", help="写入文件；省略则打印到 stdout")
    ap.add_argument("--links", action="store_true", help="文末附正文超链接列表")
    args = ap.parse_args()

    try:
        blocks, links = fetch(args.url)
    except PermissionError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    text = "\n".join(blocks)
    if args.links and links:
        text += "\n\n## 正文超链接\n" + "\n".join(
            f"- {label or '(无文字)'} → {href}" for label, href in links
        )
    if args.output:
        with open(args.output, "w", encoding="utf-8") as fh:
            fh.write(text + "\n")
        print(f"已写入 {args.output}（{len(blocks)} 块，{len(text)} 字符）", file=sys.stderr)
    else:
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
