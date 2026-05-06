#!/usr/bin/env python3
"""
fetch_to_source.py — Web 资料快速导入工具

类似 Obsidian Web Clipper，将 URL 转换为 sources/blogs/ 模板，加速 ingest。

用法：
  python3 scripts/fetch_to_source.py <URL> --name <stem>
  make fetch URL=https://... NAME=my-topic
"""

import argparse
import html
import re
import sys
from datetime import date
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

REPO_ROOT = Path(__file__).parent.parent
BLOGS_DIR = REPO_ROOT / "sources" / "blogs"


def fetch_page(url: str, timeout: int = 10) -> str:
    """抓取网页 HTML。"""
    req = Request(url, headers={"User-Agent": "Mozilla/5.0 (compatible; RoboticsNotebooks/1.0)"})
    with urlopen(req, timeout=timeout) as resp:
        charset = "utf-8"
        ct = resp.headers.get_content_charset()
        if ct:
            charset = ct
        return resp.read().decode(charset, errors="replace")


def extract_title(html_text: str, url: str) -> str:
    """从 HTML 中提取标题。"""
    # 优先 og:title
    m = re.search(r'<meta[^>]+property=["\']og:title["\'][^>]+content=["\']([^"\']+)', html_text, re.IGNORECASE)
    if m:
        return html.unescape(m.group(1).strip())
    # 次选 <title>
    m = re.search(r'<title[^>]*>([^<]+)</title>', html_text, re.IGNORECASE)
    if m:
        return html.unescape(m.group(1).strip())
    return url


def extract_description(html_text: str) -> str:
    """从 meta description 或首段提取摘要。"""
    m = re.search(
        r'<meta[^>]+name=["\']description["\'][^>]+content=["\']([^"\']{10,})', html_text, re.IGNORECASE
    )
    if m:
        return html.unescape(m.group(1).strip()[:200])
    # fallback: 第一个 <p> 段
    m = re.search(r'<p[^>]*>([^<]{20,})</p>', html_text, re.IGNORECASE)
    if m:
        return html.unescape(re.sub(r'<[^>]+>', '', m.group(1)).strip()[:200])
    return ""


def generate_template(url: str, title: str, description: str, stem: str) -> str:
    today = date.today().isoformat()
    return f"""\
# {stem}

> 来源归档（ingest）

- **标题：** {title}
- **类型：** blog
- **来源：** {url}
- **入库日期：** {today}
- **最后更新：** {today}
- **一句话说明：** {description or "（待填写）"}

## 核心摘录

### 1) {title}
- **链接：** <{url}>
- **核心要点：** （待填写）
- **对 wiki 的映射：**
  - （待填写：../../wiki/concepts/xxx.md 等）

## 当前提炼状态

- [ ] 内容摘要填写
- [ ] wiki 页面映射确认
- [ ] 关联 wiki 页面的参考来源段落已添加 ingest 链接
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="将 URL 转换为 sources/blogs/ 模板")
    parser.add_argument("url", help="目标 URL")
    parser.add_argument("--name", required=True, help="文件名 stem（不含扩展名）")
    parser.add_argument("--timeout", type=int, default=10, help="HTTP 超时秒数（默认 10）")
    args = parser.parse_args()

    print(f"正在抓取: {args.url}")
    try:
        html_text = fetch_page(args.url, args.timeout)
        title = extract_title(html_text, args.url)
        description = extract_description(html_text)
    except URLError as e:
        print(f"⚠️  抓取失败（{e}），使用 URL 作为标题生成空模板")
        title = args.url
        description = ""

    BLOGS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = BLOGS_DIR / f"{args.name}.md"
    if out_path.exists():
        print(f"❌ 文件已存在: {out_path.relative_to(REPO_ROOT)}，退出")
        sys.exit(1)

    template = generate_template(args.url, title, description, args.name)
    out_path.write_text(template, encoding="utf-8")
    print(f"✅ 生成模板: {out_path.relative_to(REPO_ROOT)}")
    print(f"   标题：{title}")
    print(f"   摘要：{description[:80]}{'...' if len(description) > 80 else ''}")
    print("\n下一步：编辑模板，填写核心要点和 wiki 映射，然后运行 make lint && make catalog")


if __name__ == "__main__":
    main()
