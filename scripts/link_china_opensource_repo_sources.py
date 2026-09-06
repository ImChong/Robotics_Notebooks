#!/usr/bin/env python3
"""为国内具身开源全景项目补齐 sources/repos/ 归档与 wiki 链接（触发 has_repo ⭐）。"""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import date
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
MAPPINGS = REPO / "exports" / "china-opensource-424-mappings.json"
URL_CACHE = REPO / "exports" / "china-opensource-424-repo-urls.json"
ENTITIES = REPO / "wiki" / "entities"
REPOS = REPO / "sources" / "repos"
BLOG = "wechat_embodied_station_domestic_opensource_panorama_2026-09-06.md"
TODAY = date.today().isoformat()

REPO_LINK_RE = re.compile(r"(?:\.\./)*sources/repos/[^)\s]+\.md\b")

MANUAL_URLS: dict[tuple[str, str], str] = {
    ("cn-os-pelican-vla-0-5", "Pelican-VLA 0.5"): "https://github.com/Open-X-Humanoid/Pelican-VLA05",
    ("cn-os-engineai-gmr", "EngineAI GMR"): "https://github.com/engineai-robotics/GMR",
    ("cn-os-sim2real", "sim2real"): "https://github.com/HighTorque-Robotics/sim2real-inference_code",
    ("cn-os-gigaworld-1", "GigaWorld-1"): "https://github.com/open-gigaai/giga-world-1",
    ("cn-os-rxbrain-1-0", "RxBrain-1.0"): "https://github.com/Tencent-Hunyuan/Hy-Embodied-RxBrain-1.0",
    ("cn-os-embodiedgen-v2", "EmbodiedGen V2"): "https://github.com/HorizonRobotics/EmbodiedGen",
    ("genie-sim-3", "Genie Sim 3.0"): "https://github.com/AgibotTech/genie_sim",
    ("agibot-world-2026", "AgiBot-World"): "https://github.com/AgibotTech/AgiBot-World",
    (
        "botworld",
        "AgiBotWorldChallengeICRA2026-WorldModelBaseline",
    ): "https://github.com/AgibotTech/AgiBotWorldChallengeICRA2026-WorldModelBaseline",
    ("botworld", "LingBot-World 2.0"): "https://github.com/Robbyant/LingBot-World",
    ("botworld", "LingBot-World 1.0"): "https://github.com/Robbyant/LingBot-World",
    ("botworld", "ABot-World"): "https://github.com/amap-cvlab/ABot-World",
    ("paper-shenlan-wm-09-gr1", "GR-1"): "https://github.com/ByteDance-Seed/GR-1",
    ("paper-opentrack", "OpenTrack / Any2Track"): "https://github.com/GalaxyGeneralRobotics/OpenTrack",
    ("paper-notebook-latent", "LATENT"): "https://github.com/GalaxyGeneralRobotics/LATENT",
}


@dataclass
class Mapping:
    layer: str
    company: str
    org_url: str
    name: str
    category: str
    slug: str
    new: bool


def parse_orgs(org_url: str) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for part in re.split(r"[、,]\s*", org_url):
        part = part.strip()
        m = re.search(r"github\.com/([^/\s]+)", part)
        if m:
            out.append(("github", m.group(1)))
        m2 = re.search(r"gitee\.com/([^/\s]+)", part)
        if m2:
            out.append(("gitee", m2.group(1)))
    return out


def repo_candidates(name: str) -> list[str]:
    base = [
        name,
        name.replace(" ", "-"),
        name.replace(" ", "_"),
        name.replace(" ", ""),
        re.sub(r"[^a-zA-Z0-9._-]+", "-", name).strip("-"),
        re.sub(r"[^a-zA-Z0-9._-]+", "_", name).strip("_"),
        name.replace(".", "-"),
        name.replace(".", "_"),
        name.replace(" 0.5", "05").replace("0.5", "05"),
        name.replace(" ", "").replace(".", ""),
        name.replace("/", "-"),
    ]
    seen: set[str] = set()
    out: list[str] = []
    for x in base:
        if x and x not in seen and not re.search(r"[\s<>\"{}|\\^`]", x):
            seen.add(x)
            out.append(x)
    return out


def head_exists(url: str) -> bool:
    req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, timeout=12) as resp:
            return resp.status in (200, 301, 302)
    except urllib.error.HTTPError as e:
        return e.code in (301, 302)
    except (urllib.error.URLError, TimeoutError):
        return False


def resolve_url(item: Mapping, cache: dict[str, str]) -> str | None:
    cache_key = f"{item.slug}\0{item.name}"
    if cache_key in cache:
        return cache[cache_key]
    manual = MANUAL_URLS.get((item.slug, item.name))
    if manual:
        cache[cache_key] = manual
        return manual
    for host, org in parse_orgs(item.org_url):
        for cand in repo_candidates(item.name):
            if host == "github":
                url = f"https://github.com/{org}/{cand}"
            else:
                url = f"https://gitee.com/{org}/{cand}"
            if head_exists(url):
                cache[cache_key] = url
                return url
    return None


def repo_filename(url: str, slug: str, name: str) -> str:
    m = re.match(r"https?://github\.com/([^/]+)/([^/]+)/?", url)
    if m:
        base = m.group(2).lower().replace(".", "_")
    else:
        m2 = re.match(r"https?://gitee\.com/([^/]+)/([^/]+)/?", url)
        if m2:
            base = m2.group(2).lower().replace(".", "_")
        elif slug.startswith("cn-os-"):
            base = slug[6:].replace("-", "_")
        else:
            base = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    base = re.sub(r"[^a-z0-9_+-]+", "_", base).strip("_")
    path = REPOS / f"{base}.md"
    if not path.exists():
        return base
    alt = slug.replace("-", "_")
    return alt if not (REPOS / f"{alt}.md").exists() else f"{alt}_{re.sub(r'[^a-z0-9]+', '', name.lower())[:20]}"


def render_repo_source(item: Mapping, url: str, wiki_slug: str) -> str:
    return f"""# {item.name}

> 来源归档（国内具身开源全景）

- **标题：** {item.name}
- **类型：** repo
- **机构：** {item.company}
- **链接：** {url}
- **分类：** {item.category}
- **入库日期：** {TODAY}
- **一句话说明：** {item.company} 开源项目 {item.name}（{item.category}），见 [国内具身开源全景](../../sources/blogs/{BLOG})。
- **沉淀到 wiki：** [`wiki/entities/{wiki_slug}.md`](../../wiki/entities/{wiki_slug}.md)

## 开源状态

- **已开源**：公开仓库（以 README 与 release 为准）。

## 对 wiki 的映射

- [wiki/entities/{wiki_slug}.md](../../wiki/entities/{wiki_slug}.md)
"""


def patch_entity(path: Path, repo_rel: str, name: str, url: str) -> bool:
    text = path.read_text(encoding="utf-8")
    if repo_rel in text:
        return False
    repo_line = f"  - ../../{repo_rel}"
    if "sources:" in text:
        text = re.sub(
            r"(sources:\n(?:  - .+\n)+)",
            lambda m: m.group(1) + (repo_line + "\n" if repo_line not in m.group(1) else ""),
            text,
            count=1,
        )
    ref_line = f"- [{name} 源码归档](../../{repo_rel})（<{url}>）"
    if "## 参考来源" in text:
        if ref_line not in text:
            text = text.replace("## 参考来源\n", f"## 参考来源\n\n{ref_line}\n", 1)
    else:
        text = text.rstrip() + f"\n\n## 参考来源\n\n{ref_line}\n"
    path.write_text(text, encoding="utf-8")
    return True


def entity_needs_repo(path: Path) -> bool:
    return not REPO_LINK_RE.search(path.read_text(encoding="utf-8"))


def load_mappings() -> list[Mapping]:
    raw = json.loads(MAPPINGS.read_text(encoding="utf-8"))
    return [Mapping(**x) for x in raw]


def load_cache() -> dict[str, str]:
    if URL_CACHE.exists():
        return json.loads(URL_CACHE.read_text(encoding="utf-8"))
    return {}


def save_cache(cache: dict[str, str]) -> None:
    URL_CACHE.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    items = load_mappings()
    cache = load_cache()

    created = 0
    patched = 0
    skipped = 0
    failed: list[tuple[str, str]] = []

    for i, item in enumerate(items):
        entity_path = ENTITIES / f"{item.slug}.md"
        if not entity_path.exists():
            continue
        if not entity_needs_repo(entity_path):
            skipped += 1
            continue
        url = resolve_url(item, cache)
        if not url:
            failed.append((item.slug, item.name))
            continue
        fname = repo_filename(url, item.slug, item.name)
        repo_path = REPOS / f"{fname}.md"
        repo_rel = f"sources/repos/{fname}.md"
        if not repo_path.exists():
            repo_path.write_text(render_repo_source(item, url, item.slug), encoding="utf-8")
            created += 1
        if patch_entity(entity_path, repo_rel, item.name, url):
            patched += 1
        if (i + 1) % 25 == 0:
            save_cache(cache)
            print(f"… {i + 1}/{len(items)} patched={patched} failed={len(failed)}")

    save_cache(cache)
    print(f"Done: repo files created={created}, entities patched={patched}, skipped={skipped}")
    if failed:
        print(f"FAILED ({len(failed)}):")
        for slug, name in failed:
            print(f"  - {slug} | {name}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
