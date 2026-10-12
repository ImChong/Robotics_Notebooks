#!/usr/bin/env python3
"""fetch_wechat_article.py — 抓取 mp.weixin.qq.com 公众号文章正文（纯文本）。

背景：WebFetch / Jina Reader / 普通 curl 访问公众号会得到「当前环境异常，完成验证后即可继续访问」
验证页。用真实 Chromium（Playwright）+ 微信内置浏览器 UA（MicroMessenger）打开则可直接读到正文
（2026-10-12 在 Claude Code 云容器实测通过）。无需 Camoufox，也不依赖 GitHub 下载。

用法：
    python3 scripts/fetch_wechat_article.py <url> [-o out.txt]

依赖：`pip install playwright`，并已有可用的 Chromium：
- 云容器已预装在 /opt/pw-browsers/chromium-*/（本脚本自动探测）；
- 或 `playwright install chromium`；
- 或用环境变量 CHROMIUM_PATH 指定可执行文件。

若输出仍含「环境异常」，说明出口 IP 被限流：等几分钟重试，或在本机用 wechat-article-for-ai 的
`--no-headless` 手动过验证后把 Markdown 交给 agent。
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

WECHAT_UA = (
    "Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) AppleWebKit/605.1.15 "
    "(KHTML, like Gecko) Mobile/15E148 MicroMessenger/8.0.50"
)
BLOCKED_MARK = "环境异常"


def find_chromium() -> str | None:
    """返回显式指定或预装的 Chromium 可执行文件；找不到则交给 Playwright 自带浏览器。"""
    env = os.environ.get("CHROMIUM_PATH")
    if env:
        return env
    root = os.environ.get("PLAYWRIGHT_BROWSERS_PATH", "/opt/pw-browsers")
    for pattern in ("chromium-*/chrome-linux*/chrome", "chromium/chrome-linux*/chrome"):
        hits = sorted(glob.glob(os.path.join(root, pattern)))
        if hits:
            return hits[-1]
    return None


def fetch(url: str, timeout_ms: int = 40000) -> str:
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path=find_chromium(), args=["--no-sandbox"])
        try:
            page = browser.new_page(user_agent=WECHAT_UA)
            page.goto(url, timeout=timeout_ms, wait_until="domcontentloaded")
            page.wait_for_timeout(3000)
            return page.inner_text("body")
        finally:
            browser.close()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("url", help="https://mp.weixin.qq.com/s/<id>")
    ap.add_argument("-o", "--output", help="写入文件；省略则打印到 stdout")
    args = ap.parse_args()

    text = fetch(args.url)
    if BLOCKED_MARK in text[:200] and len(text) < 500:
        print(
            "被微信验证页拦截（环境异常）：稍后重试，或本机 --no-headless 手动验证。",
            file=sys.stderr,
        )
        return 2
    if args.output:
        with open(args.output, "w", encoding="utf-8") as fh:
            fh.write(text)
        print(f"已写入 {args.output}（{len(text)} 字符）", file=sys.stderr)
    else:
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
