#!/usr/bin/env python3
"""Parse 国内具身开源全景 WeChat article → coverage map + stub entities."""

from __future__ import annotations

import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ENTITIES = REPO / "wiki" / "entities"
RAW_ARTICLE = Path(
    "/home/ubuntu/.cursor/projects/workspace/agent-tools/6c8d46e3-d5cc-4743-a11c-552aebaad5a3.txt"
)

LAYER_MAP = {
    "第一层": "layer1-oem",
    "第二层": "layer2-model",
    "第三层": "layer3-dexhand",
    "第四层": "layer4-platform",
    "第五层": "layer5-supply-chain",
}


@dataclass
class Project:
    layer: str
    company: str
    org_url: str
    name: str
    category: str
    description: str


def slugify(name: str) -> str:
    s = name.strip().lower()
    s = s.replace("/", "-").replace(" ", "-")
    s = re.sub(r"[^a-z0-9\u4e00-\u9fff+-]+", "-", s)
    s = re.sub(r"-+", "-", s).strip("-")
    if not s or re.fullmatch(r"[\u4e00-\u9fff]+", s):
        # fallback ascii from common replacements
        repl = {
            " ": "-",
            "_": "-",
            ".": "",
            "+": "-plus",
        }
        base = name.strip()
        for a, b in repl.items():
            base = base.replace(a, b)
        s = re.sub(r"[^a-zA-Z0-9-]+", "", base).lower()
    return s or "project"


def parse_article(text: str) -> list[Project]:
    projects: list[Project] = []
    layer = ""
    company = ""
    org_url = ""
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("### 第") and "层" in line:
            layer = line.replace("### ", "").split("·")[0].strip()
            continue
        m = re.match(r"^(.+?)（\d+ 项）｜官方组织：(.+)$", line)
        if m:
            company = m.group(1).strip()
            org_url = m.group(2).strip()
            continue
        m2 = re.match(r"^•\s*(.+?)（([^）]+)）——(.+)$", line)
        if m2 and layer and company:
            projects.append(
                Project(
                    layer=layer,
                    company=company,
                    org_url=org_url,
                    name=m2.group(1).strip(),
                    category=m2.group(2).strip(),
                    description=m2.group(3).strip(),
                )
            )
    return projects


def build_entity_index() -> dict[str, str]:
    """Map normalized keys → entity slug (without .md)."""
    index: dict[str, str] = {}
    for p in ENTITIES.glob("*.md"):
        slug = p.stem
        text = p.read_text(encoding="utf-8", errors="ignore")
        keys = {slug, slug.replace("-", ""), slug.replace("-", "_")}
        keys.add(slug.replace("-", " "))
        # h1
        for m in re.finditer(r"^#\s+(.+)$", text, re.M):
            keys.add(m.group(1).strip().lower())
        # title in frontmatter
        for m in re.finditer(r'^title:\s*["\']?(.+?)["\']?\s*$', text, re.M):
            keys.add(m.group(1).strip().lower())
        for k in keys:
            nk = normalize_key(k)
            if nk:
                index.setdefault(nk, slug)
    return index


def normalize_key(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"[^a-z0-9]+", "", s)
    return s


# Manual aliases from HMI / common naming
ALIASES: dict[str, str] = {
    "unitreerllab": "unitree-rl-lab",
    "unitreerlgym": "unitree-rl-gym",
    "unitreerlmjlab": "unitree-rl-mjlab",
    "unitreesdk2": "unitree-sdk2",
    "unitreemujoco": "unitree-mujoco",
    "unitreelerobot": "unitree-lerobot",
    "unitreemodel": "unitree-model",
    "unitreesimisaaclab": "unitree-sim-isaaclab",
    "xrteleoperate": "xr-teleoperate",
    "humanoidgym": "humanoid-gym",
    "engineairllab": "engineai-rl-lab",
    "boostergym": "paper-notebook-booster-gym-an-end-to-end-rl-framework-for-human",
    "geniesim30": "genie-sim-3",
    "geniesim": "genie-sim-3",
    "agibotworld": "agibot-world-2026",
    "opentrackany2track": "paper-opentrack",
    "opentrack": "paper-opentrack",
    "beyondmimic": "beyondmimic",
    "humanoidgpt": "paper-humanoid-gpt",
    "leggedgym": "legged-gym",
    "mimickit": "mimickit",
    "deepmimic": "deepmimic",
    "robopartytrain": "roboparty",
    "ufo": "roboparty-ufo",
    "tienkunglab": "tienkung-humanoid-open-source",
    "xiaomirobotics0": "xiaomi-robotics-0",
    "unifolmwma0": "unifolm-world-model-action",
    "unifolmvla0": "unifolm-vla",
}


def match_project(name: str, index: dict[str, str]) -> str | None:
    keys = [
        normalize_key(name),
        normalize_key(slugify(name)),
        normalize_key(name.replace(" ", "")),
        normalize_key(name.replace("-", "")),
        normalize_key(name.replace("/", "")),
    ]
    for k in keys:
        if k in ALIASES:
            return ALIASES[k]
        if k in index:
            return index[k]
    # partial: entity slug contained in project name
    nk = normalize_key(name)
    for ek, slug in index.items():
        if len(ek) >= 6 and ek in nk:
            return slug
    return None


def entity_path(slug: str) -> Path:
    return ENTITIES / f"{slug}.md"


def write_stub(proj: Project, slug: str, blog_rel: str) -> None:
    path = entity_path(slug)
    if path.exists():
        return
    today = date.today().isoformat()
    title = proj.name
    content = f"""---
type: entity
tags: [repo, china-embodied-opensource, open-source, {slugify(proj.company)[:40]}]
status: draft
updated: {today}
related:
  - ../overview/china-domestic-embodied-opensource-76-companies-technology-map.md
  - ../entities/humanoid-motion-intelligence.md
  - ../queries/china-domestic-opensource-424-coverage.md
sources:
  - ../../sources/blogs/wechat_embodied_station_domestic_opensource_panorama_2026-09-06.md
summary: "{proj.company} 开源项目 {title}（{proj.category}）：{proj.description[:120]}…"
---

# {title}

## 一句话定义

**{title}** 是 [{proj.company}]({proj.org_url.split('、')[0]}) 公开的 **{proj.category}** 开源项目：{proj.description}

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| SDK | Software Development Kit | 真机控制与状态读取接口 |
| RL | Reinforcement Learning | 强化学习训练与策略优化 |
| VLA | Vision-Language-Action | 视觉–语言–动作统一策略 |
| Sim2Real | Simulation to Real | 仿真策略迁移真机 |
| URDF | Unified Robot Description Format | 机器人描述与仿真资产 |

## 为什么重要

- 收录于 [国内具身智能开源全景（76 家 · 424 项）](../overview/china-domestic-embodied-opensource-76-companies-technology-map.md) 的 **{proj.layer}** 分组。
- 与 [Humanoid Motion Intelligence](../entities/humanoid-motion-intelligence.md) 同源策展；本页为 **独立详情节点**，便于从公司清单跳到机制与入口说明。

## 核心原理

| 字段 | 内容 |
|------|------|
| 机构 | {proj.company} |
| 类别 | {proj.category} |
| 官方组织 | {proj.org_url} |

## 工程实践

1. 从官方 GitHub/Gitee 组织检索 `{title}` 仓库并核对 README 许可与依赖。
2. 对照本库 [424 项覆盖索引](../queries/china-domestic-opensource-424-coverage.md) 查看同公司其它入口是否共用训练/部署链路。
3. 若与既有方法页（如 RL 框架、VLA、SDK）主题相同，优先读关联页中的「开源入口」小节，避免重复维护平行叙事。

## 局限与风险

- 公众号清单为 **策展快照**（2026-09-06）；仓库更名、归档或许可证变化须回官方组织页核实。
- **开源状态**：以仓库 README 与 release 为准（入库日按文章描述归纳，未逐仓 clone 验证）。

## 关联页面

- [国内具身开源全景技术地图](../overview/china-domestic-embodied-opensource-76-companies-technology-map.md)
- [HMI 开源项目主表导读](../queries/hmi-opensource-projects-coverage.md)
- [Humanoid Motion Intelligence](../entities/humanoid-motion-intelligence.md)

## 参考来源

- [国内具身智能开源全景（微信公众号）](../../sources/blogs/wechat_embodied_station_domestic_opensource_panorama_2026-09-06.md)

## 推荐继续阅读

- [{proj.company} 官方组织]({proj.org_url.split('、')[0]})
"""
    path.write_text(content, encoding="utf-8")


def main() -> int:
    if not RAW_ARTICLE.exists():
        print(f"Missing article: {RAW_ARTICLE}", file=sys.stderr)
        return 1
    text = RAW_ARTICLE.read_text(encoding="utf-8")
    projects = parse_article(text)
    print(f"Parsed {len(projects)} projects")
    index = build_entity_index()
    mappings: list[tuple[Project, str, bool]] = []
    created = 0
    reused = 0
    for p in projects:
        slug = match_project(p.name, index)
        is_new = False
        if slug is None:
            slug = f"cn-os-{slugify(p.name)}"
            # avoid collision
            base = slug
            i = 2
            while entity_path(slug).exists() and slug not in index.values():
                slug = f"{base}-{i}"
                i += 1
            write_stub(p, slug, "wechat_embodied_station_domestic_opensource_panorama_2026-09-06.md")
            if entity_path(slug).exists():
                is_new = True
                created += 1
                index[normalize_key(p.name)] = slug
        else:
            reused += 1
        mappings.append((p, slug, is_new))
    print(f"Reused {reused}, created stubs {created}")
    # write coverage json for markdown gen
    out = REPO / "exports" / "china-opensource-424-mappings.json"
    import json

    out.write_text(
        json.dumps(
            [
                {
                    "layer": m[0].layer,
                    "company": m[0].company,
                    "org_url": m[0].org_url,
                    "name": m[0].name,
                    "category": m[0].category,
                    "slug": m[1],
                    "new": m[2],
                }
                for m in mappings
            ],
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
