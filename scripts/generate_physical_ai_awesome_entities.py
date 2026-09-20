#!/usr/bin/env python3
"""Deepen natnew + aichr awesome-physical-ai into independent wiki nodes.

Parses both curated READMEs, unions/dedupes by arXiv / GitHub / URL / title,
reuses existing canonical wiki pages, and writes index-level entities for
missing items plus a technology-map hub.

Idempotent: re-running skips existing pai-* / paper-pai-* files and never
creates a second frontmatter arXiv ID.
"""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from collections import defaultdict
from datetime import date
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
TODAY = date.today().isoformat()
CACHE_DIR = Path("/tmp/pai-awesome")
TECH_MAP = "wiki/overview/awesome-physical-ai-technology-map.md"
CATALOG = "sources/repos/awesome-physical-ai-union-catalog.md"
STATS_PATH = CACHE_DIR / "pai_awesome_gen_stats.json"

NATNEW_URL = "https://github.com/natnew/awesome-physical-ai"
AICHR_URL = "https://github.com/aichr/awesome-physical-ai"
NATNEW_ENTITY = "wiki/entities/awesome-physical-ai-natnew.md"
AICHR_ENTITY = "wiki/entities/awesome-physical-ai-aichr.md"
COMPARE_ENTITY = "wiki/comparisons/awesome-physical-ai-curated-lists.md"

ARXIV_RE = re.compile(
    r"(?:arxiv\.org/(?:abs|html|pdf)/|arxiv:\s*)(\d{4}\.\d{4,5})(?:v\d+)?",
    re.I,
)
GH_RE = re.compile(r"github\.com/([^/\s\"')]+)/([^/\s\"')]+)", re.I)
HF_RE = re.compile(
    r"huggingface\.co/(?:datasets/)?([^/\s\"')]+)/([^/\s\"')]+)",
    re.I,
)

SKIP_SECTIONS = {
    "contents",
    "contributing",
    "legend",
    "star history",
    "get started",
    "choose your route",
    "quick start path",
    "appendices",
}

# Title / slug → existing canonical page (ambiguous or non-exact H1 cases).
HAND_ALIASES: dict[str, str] = {
    "mujoco": "wiki/entities/mujoco.md",
    "nvidia isaac sim": "wiki/entities/isaac-sim.md",
    "isaac sim": "wiki/entities/isaac-sim.md",
    "isaac lab": "wiki/entities/isaac-lab.md",
    "isaac lab documentation": "wiki/entities/isaac-lab.md",
    "isaac gym": "wiki/entities/isaac-gym.md",
    "drake": "wiki/entities/drake.md",
    "gazebo": "wiki/entities/gazebo-sim.md",
    "pybullet": "wiki/entities/pybullet.md",
    "pybullet quickstart": "wiki/entities/pybullet.md",
    "habitat": "wiki/entities/habitat-sim.md",
    "sapien": "wiki/entities/sapien.md",
    "genesis": "wiki/entities/genesis-sim.md",
    "webots": "wiki/entities/webots.md",
    "coppeliasim": "wiki/entities/coppeliasim.md",
    "carla": "wiki/entities/carla.md",
    "airsim": "wiki/entities/airsim.md",
    "brax": "wiki/entities/brax.md",
    "gymnasium": "wiki/entities/gymnasium.md",
    "lerobot": "wiki/entities/lerobot.md",
    "lerobot tutorials": "wiki/entities/lerobot.md",
    "lerobot evaluation scripts": "wiki/entities/lerobot.md",
    "lerobot hardware": "wiki/entities/lerobot.md",
    "lerobot tutorial": "wiki/entities/lerobot.md",
    "openvla": "wiki/entities/openvla.md",
    "open x-embodiment": "wiki/concepts/open-x-embodiment.md",
    "open x-embodiment tutorial": "wiki/concepts/open-x-embodiment.md",
    "rt-x": "wiki/entities/paper-open-x-embodiment.md",
    "libero": "wiki/entities/libero-benchmark.md",
    "rlbench": "wiki/entities/rlbench.md",
    "calvin": "wiki/entities/calvin-benchmark.md",
    "calvin abc-d": "wiki/entities/calvin-benchmark.md",
    "maniskill": "wiki/entities/maniskill2.md",
    "maniskill benchmark": "wiki/entities/maniskill2.md",
    "diffusion policy": "wiki/methods/diffusion-policy.md",
    "diffusion policy tutorial": "wiki/methods/diffusion-policy.md",
    "octo": "wiki/entities/paper-octo.md",
    "rt-2": "wiki/entities/paper-rt-2.md",
    "rt-1": "wiki/entities/paper-rt-1.md",
    "rt-1 dataset": "wiki/entities/paper-rt-1.md",
    "π0 (physical intelligence)": "wiki/entities/paper-pi0.md",
    "physical intelligence π0": "wiki/entities/paper-pi0.md",
    "physical intelligence π0.5": "wiki/entities/paper-pi05-open-world-vla.md",
    "palm-e": "wiki/entities/paper-palm-e-embodied-language-model.md",
    "gemini robotics": "wiki/entities/gemini-robotics.md",
    "figure": "wiki/entities/figure-ai.md",
    "1x technologies": "wiki/entities/1x-technologies.md",
    "boston dynamics": "wiki/entities/boston-dynamics.md",
    "boston dynamics atlas": "wiki/entities/boston-dynamics.md",
    "boston dynamics spot": "wiki/entities/boston-dynamics.md",
    "boston dynamics blog": "wiki/entities/boston-dynamics.md",
    "unitree robotics": "wiki/entities/unitree.md",
    "unitree h1/g1": "wiki/entities/unitree-g1.md",
    "unitree go2/b2": "wiki/entities/unitree.md",
    "ros 2": "wiki/concepts/ros2-basics.md",
    "ros 2 tutorials": "wiki/concepts/ros2-basics.md",
    "ros 2 control": "wiki/entities/ros2-control.md",
    "ros2_control": "wiki/entities/ros2-control.md",
    "moveit 2": "wiki/entities/moveit2.md",
    "foxglove": "wiki/entities/foxglove-studio.md",
    "aloha hardware": "wiki/entities/aloha.md",
    "act (action chunking transformers)": "wiki/entities/aloha.md",
    "domain randomization (tobin et al.)": "wiki/concepts/domain-randomization.md",
    "domain randomization": "wiki/concepts/domain-randomization.md",
    "nvidia cosmos": "wiki/entities/cosmos-3.md",
    "modern robotics": "wiki/entities/modern-robotics-book.md",
    "robocasa": "wiki/entities/robocasa.md",
    "behavior-1k": "wiki/entities/behavior-1k.md",
    "awesome-touch": "wiki/entities/awesome-touch.md",
    "awesome touch": "wiki/entities/awesome-touch.md",
    "awesome world models": "wiki/entities/awesome-world-models.md",
    "mujoco menagerie": "wiki/entities/mujoco-menagerie.md",
    "mujoco documentation": "wiki/entities/mujoco.md",
    "droid": "wiki/entities/droid-policy-learning.md",
    "nvidia gr00t": "wiki/entities/isaac-gr00t.md",
    "gr00t n1 (nvidia)": "wiki/entities/isaac-gr00t.md",
    "lingbot-vla": "wiki/entities/lingbot-vla.md",
    "daydreamer": "wiki/entities/paper-daydreamer-world-models-real-robots.md",
    "dreamerv3": "wiki/entities/paper-shenlan-wm-13-dreamerv3.md",
    "v-jepa 2 (meta fair)": "wiki/entities/paper-vjepa2.md",
    "v-jepa 2": "wiki/entities/paper-vjepa2.md",
    "unisim": "wiki/entities/paper-unisim.md",
    "td-mpc2": "wiki/entities/paper-td-mpc2.md",
    "planet": "wiki/entities/paper-planet-latent-dynamics.md",
    "muzero": "wiki/entities/paper-muzero-planning-latent-dynamics.md",
    "gaia-1 (wayve)": "wiki/entities/paper-gaia1.md",
    "saycan": "wiki/methods/saycan.md",
    "igibson": "wiki/entities/igibson.md",
    "raisim": "wiki/entities/raisim.md",
    "robosuite": "wiki/entities/robosuite.md",
    "anygrasp": "wiki/entities/anygrasp.md",
    "deepmimic": "wiki/methods/deepmimic.md",
    "rsl-rl": "wiki/entities/rsl-rl.md",
    "learning to walk in minutes (eth/rsl)": "wiki/entities/legged-gym.md",
    "hover — versatile humanoid whole-body controller": "wiki/entities/paper-bfm-14-hover.md",
    "hover: versatile neural whole-body controller": "wiki/entities/paper-bfm-14-hover.md",
    "asap — sim-to-real for humanoid whole-body skills": "wiki/entities/paper-notebook-asap-aligning-simulation-and-real-world-physics.md",
    "asap: aligning simulation and real-world physics": "wiki/entities/paper-notebook-asap-aligning-simulation-and-real-world-physics.md",
    "control barrier functions": "wiki/concepts/control-barrier-function.md",
    "eclipse cyclone dds": "wiki/entities/cyclone-dds.md",
    "fast dds": "wiki/entities/fast-dds.md",
    "mcap": "wiki/entities/mcap-log-format.md",
    "franka emika": "wiki/entities/franka-research-3.md",
    "fourier intelligence gr-1": "wiki/entities/fourier-grx-n1.md",
    "anymal parkour (rsl)": "wiki/entities/anymal.md",
    "anybotics": "wiki/entities/anymal.md",
    "reachy 2 (pollen robotics / hugging face)": "wiki/entities/pollen-reachy2.md",
    "pollen robotics (hugging face)": "wiki/entities/pollen-reachy2.md",
    "tesla optimus": "wiki/entities/tesla-optimus.md",
    "skild ai": "wiki/entities/skild-ai.md",
    "nvidia jetson": "wiki/entities/nvidia-jetson.md",
    "tensorrt": "wiki/entities/tensorrt.md",
    "helix (figure)": "wiki/entities/helix-25.md",
    "omnih2o": "wiki/entities/paper-hrl-stack-08-omnih2o.md",
    "humanplus": "wiki/entities/paper-loco-manip-161-012-humanplus.md",
    "walk these ways": "wiki/entities/paper-walk-these-ways-quadruped-mob.md",
    "robomimic": "wiki/entities/robomimic.md",
    "roboarena": "wiki/methods/roboarena.md",
    "metaworld": "wiki/entities/paper-hrl-stack-32-metaworld.md",
    "mobile aloha": "wiki/entities/aloha.md",
    "isaac ros": "wiki/entities/isaac-ros-nvblox.md",
    "nvidia isaac ros": "wiki/entities/isaac-ros-nvblox.md",
}

SECTION_KIND: dict[str, str] = {
    "simulators": "simulator",
    "datasets": "dataset",
    "benchmarks": "benchmark",
    "evaluation methodology": "eval",
    "robotics foundation models": "paper",
    "foundation models (vla)": "paper",
    "world models": "paper",
    "manipulation": "method",
    "locomotion": "paper",
    "sim-to-real": "paper",
    "safety & robustness": "resource",
    "governance & policy": "standard",
    "production patterns / reference architectures": "resource",
    "courses": "course",
    "courses & tutorials": "course",
    "companies": "company",
    "books": "book",
    "tutorials & guides": "guide",
    "key papers": "paper",
    "survey papers": "paper",
    "hardware platforms": "hardware",
    "hardware & actuation": "hardware",
    "conferences": "conference",
    "community": "community",
    "newsletters & blogs": "community",
    "people to follow": "person",
    "related awesome lists": "list",
    "3d computer vision": "resource",
    "edge ai & inference": "resource",
    "frameworks & libraries": "resource",
    "robot platforms": "hardware",
    "research labs": "lab",
}

KIND_ABBREV: dict[str, list[tuple[str, str, str]]] = {
    "paper": [
        ("PAI", "Physical AI", "具身/物理智能策展主题"),
        ("VLA", "Vision-Language-Action", "视觉–语言–动作策略"),
        ("RFM", "Robotics Foundation Model", "机器人基础模型"),
        ("Sim2Real", "Simulation to Real", "仿真到真机迁移"),
    ],
    "dataset": [
        ("PAI", "Physical AI", "具身/物理智能策展主题"),
        ("OXE", "Open X-Embodiment", "跨本体轨迹语料参照"),
        ("IL", "Imitation Learning", "演示数据驱动的模仿学习"),
    ],
    "simulator": [
        ("PAI", "Physical AI", "具身/物理智能策展主题"),
        ("Sim2Real", "Simulation to Real", "仿真策略迁移到真机"),
        ("RL", "Reinforcement Learning", "仿真里最常见的训练回路"),
    ],
    "benchmark": [
        ("PAI", "Physical AI", "具身/物理智能策展主题"),
        ("VLA", "Vision-Language-Action", "语言条件操作评测主线"),
        ("RL", "Reinforcement Learning", "任务套件常用训练范式"),
    ],
    "company": [
        ("PAI", "Physical AI", "具身/物理智能策展主题"),
        ("VLA", "Vision-Language-Action", "工业通才策略常见形态"),
        ("FM", "Foundation Model", "跨本体基础模型产品线"),
    ],
    "course": [
        ("PAI", "Physical AI", "具身/物理智能策展主题"),
        ("RL", "Reinforcement Learning", "课程主干算法族"),
        ("IL", "Imitation Learning", "演示学习对照路径"),
    ],
    "book": [
        ("PAI", "Physical AI", "具身/物理智能策展主题"),
        ("SLAM", "Simultaneous Localization and Mapping", "经典感知定位教材主题"),
        ("MPC", "Model Predictive Control", "规划/控制教材主题"),
    ],
    "conference": [
        ("CoRL", "Conference on Robot Learning", "机器人学习主会"),
        ("RSS", "Robotics: Science and Systems", "机器人科学主会"),
        ("ICRA", "IEEE International Conference on Robotics and Automation", "机器人自动化主会"),
    ],
    "person": [
        ("PAI", "Physical AI", "具身/物理智能策展主题"),
        ("VLA", "Vision-Language-Action", "该领域研究者常见方向"),
        ("RL", "Reinforcement Learning", "策略学习研究主线"),
    ],
    "community": [
        ("PAI", "Physical AI", "具身/物理智能策展主题"),
        ("ROS", "Robot Operating System", "社区讨论高频中间件"),
        ("VLA", "Vision-Language-Action", "开源具身讨论主线"),
    ],
    "standard": [
        ("PAI", "Physical AI", "具身/物理智能策展主题"),
        ("RMF", "Risk Management Framework", "AI 风险管理框架"),
        ("ISO", "International Organization for Standardization", "机器人/功能安全标准族"),
    ],
    "hardware": [
        ("PAI", "Physical AI", "具身/物理智能策展主题"),
        ("DoF", "Degrees of Freedom", "臂/人形自由度口径"),
        ("ROS", "Robot Operating System", "常用驱动与中间件"),
    ],
    "lab": [
        ("PAI", "Physical AI", "具身/物理智能策展主题"),
        ("VLA", "Vision-Language-Action", "实验室常见研究主题"),
        ("RL", "Reinforcement Learning", "策略学习主线"),
    ],
    "list": [
        ("PAI", "Physical AI", "具身/物理智能策展主题"),
        ("VLA", "Vision-Language-Action", "相邻清单高频主题"),
        ("WM", "World Model", "世界模型相邻清单"),
    ],
    "eval": [
        ("PAI", "Physical AI", "具身/物理智能策展主题"),
        ("RL", "Reinforcement Learning", "评测统计口径常用于 RL"),
        ("VLA", "Vision-Language-Action", "通才策略评测对象"),
    ],
    "guide": [
        ("PAI", "Physical AI", "具身/物理智能策展主题"),
        ("IL", "Imitation Learning", "教程常见训练路径"),
        ("RL", "Reinforcement Learning", "入门回路"),
    ],
    "method": [
        ("PAI", "Physical AI", "具身/物理智能策展主题"),
        ("IL", "Imitation Learning", "操作方法主线"),
        ("VLA", "Vision-Language-Action", "语言条件操作"),
    ],
    "resource": [
        ("PAI", "Physical AI", "具身/物理智能策展主题"),
        ("VLA", "Vision-Language-Action", "资源条目常见落点"),
        ("Sim2Real", "Simulation to Real", "部署与迁移对照"),
    ],
}

SELF_REPOS = {
    "natnew/awesome-physical-ai",
    "aichr/awesome-physical-ai",
}


def _slugify(title: str, max_len: int = 42) -> str:
    s = unicodedata.normalize("NFKD", title)
    s = s.encode("ascii", "ignore").decode("ascii")
    s = s.lower()
    s = re.sub(r"[^a-z0-9]+", "-", s).strip("-")
    return (s or "untitled")[:max_len].strip("-")


def _norm_title(title: str) -> str:
    s = title.lower()
    s = re.sub(r"\([^)]*\)", " ", s)
    s = unicodedata.normalize("NFKD", s)
    s = s.encode("ascii", "ignore").decode("ascii")
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def _yaml_escape(s: str) -> str:
    s = s.replace("\\", "\\\\").replace('"', "'").replace("\n", " ").strip()
    s = re.sub(r"https?://\S+", "", s)
    s = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", s)
    s = re.sub(r"\s+", " ", s).strip(" -|")
    return s


def _yaml_list(items: list[str], indent: int = 2) -> str:
    pad = " " * indent
    return "\n".join(f"{pad}- {x}" for x in items)


def _clean_section(raw: str) -> str:
    s = re.sub(r"<[^>]+>", "", raw)
    s = re.sub(r"[#*]+", "", s).strip()
    return s


def _extract_arxiv(text: str) -> str | None:
    m = ARXIV_RE.search(text)
    return m.group(1) if m else None


def _extract_github(text: str) -> str | None:
    for m in GH_RE.finditer(text):
        owner, repo = m.group(1).lower(), m.group(2).lower()
        repo = repo.removesuffix(".git").rstrip(".,);")
        if owner in {"topics", "orgs", "settings"}:
            continue
        key = f"{owner}/{repo}"
        if key in SELF_REPOS:
            continue
        return key
    return None


def _normalize_url(url: str) -> str:
    url = url.strip().rstrip(").,;")
    p = urlparse(url)
    host = (p.netloc or "").lower().removeprefix("www.")
    path = p.path.rstrip("/")
    if host == "arxiv.org":
        aid = _extract_arxiv(url)
        return f"arxiv:{aid}" if aid else url
    if host == "github.com":
        gh = _extract_github(url)
        return f"gh:{gh}" if gh else f"url:{host}{path.lower()}"
    return f"url:{host}{path.lower()}"


def _kind_for(section: str, arxiv: str | None) -> str:
    key = section.lower().strip()
    kind = SECTION_KIND.get(key, "resource")
    if arxiv and kind in {"method", "resource", "eval"}:
        return "paper"
    return kind


def _is_entry_line(line: str) -> re.Match[str] | None:
    return re.match(r"^[\-\+]\s+(.*)$", line)


def parse_readme(path: Path, list_key: str) -> list[dict[str, Any]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    entries: list[dict[str, Any]] = []
    cur_sec = "ROOT"
    i = 0
    while i < len(lines):
        line = lines[i]
        hm = re.match(r"^(#{1,4})\s+(.+)$", line)
        if hm:
            title = _clean_section(hm.group(2))
            if title.lower() in SKIP_SECTIONS or title.lower().startswith("awesome physical"):
                cur_sec = "SKIP"
            elif hm.group(1) == "#":
                cur_sec = title
            elif hm.group(1) == "##":
                cur_sec = title
            i += 1
            continue
        if cur_sec in {"SKIP", "ROOT"}:
            i += 1
            continue
        em = _is_entry_line(line)
        if not em:
            i += 1
            continue
        blob_parts = [line]
        i += 1
        while i < len(lines):
            nxt = lines[i]
            if re.match(r"^#{1,4}\s+", nxt) or _is_entry_line(nxt):
                break
            if nxt.strip().startswith("<!--"):
                blob_parts.append(nxt)
                i += 1
                continue
            if nxt.strip() == "":
                break
            blob_parts.append(nxt)
            i += 1
        blob = " ".join(p.strip() for p in blob_parts if p.strip())
        if cur_sec.lower() in SKIP_SECTIONS:
            continue
        first = blob_parts[0].strip()
        md_start = re.match(
            r"^[\-\+]\s+\[([^\]]+)\]\((https?://[^)]+)\)",
            first,
        )
        bold_start = re.match(r"^[\-\+]\s+\*\*([^*]+)\*\*", first)
        GENERIC = {
            "github",
            "code",
            "paper",
            "site",
            "docs",
            "page",
            "blog",
            "article",
            "reddit",
            "link",
            "pdf",
            "arxiv",
            "website",
        }
        if md_start:
            title = md_start.group(1).strip()
            url = md_start.group(2).rstrip(").,;")
        elif bold_start:
            title = bold_start.group(1).strip()
            um = re.search(r"(https?://[^\s)]+)", blob)
            url = um.group(1).rstrip(").,;") if um else ""
        else:
            md = re.search(r"\[([^\]]+)\]\((https?://[^)]+)\)", blob)
            if md:
                title = md.group(1).strip()
                url = md.group(2).rstrip(").,;")
            else:
                title = re.sub(r"^[\-\+]\s+", "", first)
                title = re.sub(r"https?://\S+", "", title)
                title = re.sub(r"[📄💻🤖🎬📊]+", "", title).strip(" -—:")
                um = re.search(r"(https?://[^\s)]+)", blob)
                url = um.group(1).rstrip(").,;") if um else ""
        if title.lower() in GENERIC:
            continue
        title = re.sub(r"[📄💻🤖🎬📊]+", "", title).strip()
        if not title or len(title) < 2:
            continue
        if title.lower() in {"contributing", "contents", "license", "star history"}:
            continue
        desc = blob
        desc = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", desc)
        desc = re.sub(r"https?://\S+", "", desc)
        desc = re.sub(r"[📄💻🤖🎬📊]+", "", desc)
        desc = re.sub(r"^[\-\+]\s+", "", desc)
        desc = re.sub(re.escape(title), "", desc, count=1, flags=re.I)
        desc = re.sub(
            r"\b(Code|GitHub|Site|Paper|Docs|Page|Blog|Article|Reddit|Website|PDF|Link)\b",
            "",
            desc,
        )
        desc = re.sub(r"^[\s*_:—\-–]+", "", desc)
        desc = re.sub(r"\s+", " ", desc).strip(" -—:*")
        if len(desc) > 280:
            desc = desc[:277].rstrip() + "..."
        extra_urls = re.findall(r"\((https?://[^)]+)\)", blob)
        arxiv = _extract_arxiv(blob + " " + " ".join(extra_urls))
        github = _extract_github(blob + " " + " ".join([url, *extra_urls]))
        if github and github in SELF_REPOS:
            continue
        if not url and not arxiv and not github:
            continue
        entries.append(
            {
                "title": title,
                "section": cur_sec,
                "list": list_key,
                "url": url,
                "extra_urls": extra_urls,
                "desc": desc or f"{list_key} 清单收录：{title}",
                "arxiv": arxiv,
                "github": github,
                "blob": blob[:500],
            }
        )
    return entries


def _stable_id(e: dict[str, Any]) -> str:
    if e.get("arxiv"):
        return f"arxiv:{e['arxiv']}"
    if e.get("github"):
        return f"gh:{e['github']}"
    if e.get("url"):
        return _normalize_url(e["url"])
    return f"title:{_norm_title(e['title'])}"


def union_entries(raw: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_id: dict[str, dict[str, Any]] = {}
    title_to_id: dict[str, str] = {}
    for e in raw:
        sid = _stable_id(e)
        nt = _norm_title(e["title"])
        # Merge same normalized title when neither side has a conflicting arxiv.
        if nt in title_to_id:
            other_id = title_to_id[nt]
            other = by_id[other_id]
            other_arxiv = other.get("arxiv")
            if e.get("arxiv") and other_arxiv and e["arxiv"] != other_arxiv:
                pass
            else:
                sid = other_id
        if sid in by_id:
            cur = by_id[sid]
            if e["list"] not in cur["lists"]:
                cur["lists"].append(e["list"])
            if e["section"] not in cur["sections"]:
                cur["sections"].append(e["section"])
            if e.get("arxiv") and not cur.get("arxiv"):
                cur["arxiv"] = e["arxiv"]
            if e.get("github") and not cur.get("github"):
                cur["github"] = e["github"]
            if e.get("url") and not cur.get("url"):
                cur["url"] = e["url"]
            if len(e.get("desc", "")) > len(cur.get("desc", "")):
                cur["desc"] = e["desc"]
            continue
        item = {
            **e,
            "sid": sid,
            "lists": [e["list"]],
            "sections": [e["section"]],
            "kind": _kind_for(e["section"], e.get("arxiv")),
            "norm": nt,
        }
        by_id[sid] = item
        title_to_id.setdefault(nt, sid)
    items = list(by_id.values())
    items.sort(key=lambda x: (x["sections"][0].lower(), x["title"].lower()))
    return items


def _frontmatter_block(text: str) -> str:
    m = re.match(r"^---\n(.*?)\n---", text, re.DOTALL)
    return m.group(1) if m else ""


class WikiIndex:
    def __init__(self) -> None:
        self.arxiv: dict[str, str] = {}
        self.github: dict[str, str] = {}
        self.title: dict[str, str] = {}
        self.stem: dict[str, str] = {}

    def load(self) -> None:
        for folder in ("entities", "concepts", "methods", "comparisons", "overview"):
            d = ROOT / "wiki" / folder
            if not d.exists():
                continue
            for p in d.glob("*.md"):
                if p.name.lower() in {"readme.md", "index.md"}:
                    continue
                rel = str(p.relative_to(ROOT))
                # Head-only: full wiki bodies are huge and not needed for matching.
                head = p.read_text(encoding="utf-8", errors="ignore")[:4000]
                fm = _frontmatter_block(head)
                am = re.search(r'(?m)^arxiv:\s*["\']?(\d{4}\.\d{4,5})', fm)
                if am:
                    self.arxiv.setdefault(am.group(1), rel)
                scan = fm + "\n" + head
                gh = _extract_github(scan)
                if gh:
                    self.github.setdefault(gh, rel)
                hm = re.search(r"^#\s+(.+)$", head, re.M)
                if hm:
                    self.title.setdefault(_norm_title(hm.group(1)), rel)
                stem = p.stem.lower()
                self.stem.setdefault(stem, rel)
                for prefix in ("paper-", "dataset-", "awesome-"):
                    if stem.startswith(prefix):
                        self.stem.setdefault(stem[len(prefix) :], rel)


def resolve_existing(e: dict[str, Any], idx: WikiIndex) -> str | None:
    if e.get("arxiv") and e["arxiv"] in idx.arxiv:
        return idx.arxiv[e["arxiv"]]
    alias_keys = [_norm_title(e["title"]), e["title"].lower().strip()]
    for k in alias_keys:
        if k in HAND_ALIASES:
            return HAND_ALIASES[k]
    if e.get("github") and e["github"] in idx.github:
        return idx.github[e["github"]]
    nt = e["norm"]
    if nt in idx.title:
        return idx.title[nt]
    slug = _slugify(e["title"], 48)
    if slug in idx.stem:
        return idx.stem[slug]
    return None


def _compact_slug(title: str, max_len: int = 36) -> str:
    """Hyphen-free slug so filenames never match TOOL_NAME_HINTS ('-lab', '-sim')."""
    slug = _slugify(title, max_len).replace("-", "")
    # Avoid '{seq}-{ros...}' / '{seq}-{sim...}' matching TOOL_NAME_HINTS.
    if slug.startswith(("sim", "gym", "lab", "ros", "sdk", "lib", "slam", "vio")):
        slug = "x" + slug
    return slug


def new_wiki_rel(e: dict[str, Any], seq: int) -> str:
    slug = _compact_slug(e["title"])
    if e.get("arxiv") and e["kind"] == "paper":
        a = e["arxiv"].replace(".", "-")
        return f"wiki/entities/paper-pai-{a}-{slug}.md"
    return f"wiki/entities/painode-{seq:03d}-{slug or 'item'}.md"


def new_source_rel(e: dict[str, Any], seq: int) -> str:
    slug = _slugify(e["title"], 36)
    if e.get("arxiv"):
        a = e["arxiv"].replace(".", "_")
        return f"sources/papers/pai_awesome_{a}_{slug}.md"
    return f"sources/repos/pai_awesome_{e['kind']}_{seq:03d}_{slug}.md"


def wiki_rel_from_root(path: str) -> str:
    assert path.startswith("wiki/")
    return "../" + path[len("wiki/") :]


def _hubs_for(e: dict[str, Any]) -> list[str]:
    sec = " ".join(e["sections"]).lower()
    out = [
        wiki_rel_from_root(NATNEW_ENTITY)
        if "natnew" in e["lists"]
        else wiki_rel_from_root(AICHR_ENTITY),
        "../overview/awesome-physical-ai-technology-map.md",
    ]
    if "loco" in sec:
        out += ["../tasks/locomotion.md", "../concepts/sim2real.md"]
    elif "manip" in sec or "dataset" in sec or "vla" in sec or "foundation" in sec:
        out += ["../methods/vla.md", "../tasks/manipulation.md"]
    elif "world" in sec:
        out += ["../methods/generative-world-models.md", "../methods/vla.md"]
    elif "sim" in sec:
        out += ["../concepts/sim2real.md", "../tasks/locomotion.md"]
    elif "safety" in sec or "govern" in sec:
        out += ["../concepts/sim2real.md", "../methods/vla.md"]
    else:
        out += ["../methods/vla.md", "../concepts/sim2real.md"]
    seen: set[str] = set()
    uniq: list[str] = []
    for x in out:
        if x not in seen:
            seen.add(x)
            uniq.append(x)
    return uniq[:6]


def render_source(e: dict[str, Any], wiki_rel: str, idx: int, total: int) -> str:
    lists = "、".join(e["lists"])
    arxiv_line = (
        f"- **arXiv：** {e['arxiv']}" if e.get("arxiv") else "- **arXiv：** （无 / 非 arXiv）"
    )
    gh_line = (
        f"- **代码：** <https://github.com/{e['github']}>"
        if e.get("github")
        else "- **代码：** 未在清单中标注"
    )
    url_line = f"- **主链接：** <{e['url']}>" if e.get("url") else ""
    return f"""# {e["title"]}

> 来源归档（awesome-physical-ai 策展索引级）

- **列表：** {lists}（[natnew]({NATNEW_URL}) / [aichr]({AICHR_URL})）
- **分组：** {", ".join(e["sections"])}
- **编号：** {idx:03d}/{total:03d}
- **入库日期：** {TODAY}
{arxiv_line}
{url_line}
{gh_line}
- **清单摘要：** {e["desc"]}
- **沉淀到 wiki：** [`{wiki_rel}`](../../{wiki_rel})

---

## 开源边界（步骤 2.5）

| 已发布 | 备注 |
|--------|------|
| 清单条目元数据 | 本 source 为策展摘录，非全文转存 |
| 代码/权重 | 以项目页 / GitHub 实际链接为准；清单标注见上 |

## 对 wiki 的映射

- 实体页：[`{wiki_rel}`](../../{wiki_rel})
- 列表实体：[natnew](../../{NATNEW_ENTITY}) · [aichr](../../{AICHR_ENTITY})
- 技术地图：[`{TECH_MAP}`](../../{TECH_MAP})
"""


def render_entity(e: dict[str, Any], src_rel: str, idx: int, total: int) -> str:
    short = e["title"].split(":")[0].strip() if ":" in e["title"] else e["title"]
    if len(short) > 80:
        short = short[:77] + "..."
    summary = _yaml_escape(e["desc"][:220]) or _yaml_escape(e["title"])
    tags = ["curated-index", "physical-ai", "awesome-physical-ai"]
    kind_tag = e["kind"]
    if kind_tag in {
        "software",
        "repo",
        "framework",
        "simulator",
        "library",
        "sdk",
        "middleware",
        "tool",
        "tooling",
        "package",
        "api",
        "engine",
        "suite",
        "toolkit",
        "firmware",
    }:
        kind_tag = "sim-env"
    tags.append(kind_tag)
    if e.get("arxiv"):
        tags.append("paper")
    related = _hubs_for(e)
    fm_extra: list[str] = []
    if e.get("arxiv"):
        fm_extra.append(f'arxiv: "{e["arxiv"]}"')
    if e.get("github"):
        fm_extra.append(f"code: https://github.com/{e['github']}")
    fm_extra_s = ("\n".join(fm_extra) + "\n") if fm_extra else ""
    abbrev = KIND_ABBREV.get(e["kind"], KIND_ABBREV["resource"])
    abbrev_rows = "\n".join(f"| {a} | {b} | {c} |" for a, b, c in abbrev[:4])
    primary = e.get("url") or (
        f"https://arxiv.org/abs/{e['arxiv']}" if e.get("arxiv") else NATNEW_URL
    )
    lists = "、".join(e["lists"])
    sections = " / ".join(e["sections"])
    gh_row = f"\n| 代码/仓库 | <https://github.com/{e['github']}> |" if e.get("github") else ""
    arxiv_row = f"\n| arXiv | {e['arxiv']} |" if e.get("arxiv") else ""
    is_paper = e["kind"] == "paper" or bool(e.get("arxiv"))
    paper_sections = ""
    if is_paper:
        paper_sections = f"""
## 核心机制（归纳）

### 策展导读要点

{e["desc"]}

本页不复述论文公式与完整实验表；工程落地请回到原文 / 项目页，并对照站内 VLA、Sim2Real 或任务页。

## 评测与指标（索引级）

- 本条目为 Awesome 策展 **索引级** 摘录，**未搬运** 原文量化 benchmark 与实机指标。
- 评测口径与具体数值以 [原文 / 项目页]({primary}) 为准。
- 横向对照请回到 [技术地图](../overview/awesome-physical-ai-technology-map.md) 同分组条目。

## 与其他工作对比（索引级）

- 本页 **不做** 与具体基线的逐项数值对比：索引级节点只保留清单坐标。
- 与站内 **深度论文实体** 的分界：深度页承载机构、实验表与源码运行时序；本页只承载清单导读锚点。同一 arXiv 若已存在深度页，应以深度页为准。
- 两份同名清单可能对同一工作给出不同链接（项目页 / arXiv / GitHub）；canonical 以本页 frontmatter 与技术地图为准。

## 结论

**本条目的站内价值是把「{short}」从 awesome-physical-ai 列表提升为可链接的知识节点，并保留清单分组作为阅读锚点。**

1. **策展坐标** — 分组 **{sections}**，来源 {lists}。
2. **适用边界** — 索引级页面不能替代 PDF / 官方文档；开源状态以项目页实际链接为准。
3. **去重** — 同一 arXiv / 同一 GitHub 仓在全库只允许一个 canonical 详情节点。
4. **升格条件** — 若该工作进入学习主线，再补机构、实验表与源码运行时序图。

## 源码运行时序图

**不适用**（索引级节点未逐仓核 README 训练/推理入口；清单标注的仓库链接可能滞后，复现前按 ingest 步骤 2.5 打开项目页核对）。
"""
    else:
        paper_sections = f"""
## 核心原理

{e["desc"]}

该条目在 Physical AI 清单中的角色是 **{e["kind"]}**，分组 **{sections}**。本页只固化清单给出的问题设定与入口链接，不把外部营销页或课程大纲转存成知识正文。

输入是读者要从清单跳到可复核的官方入口；输出是站内可检索、可互链的详情节点。机制细节、API 与版本以官方文档为准。

## 工程实践

| 字段 | 内容 |
|------|------|
| 官方入口 | <{primary}> |{gh_row}
| 开源核查 | 以项目页 / GitHub 实际链接为准（清单可能滞后） |
| 源码运行时序图 | **不适用**（非论文可运行训练仓，或未核 README 入口） |

调试时先确认链接指向的是官方仓/文档而不是镜像或过期 fork，再决定是否升格为深度实体页。
"""
    src_links = [
        f"- [`{src_rel}`](../../{src_rel}) — 本条目策展摘录",
        f"- [`{CATALOG}`](../../{CATALOG}) — 双清单并集目录",
        "- [sources/repos/awesome-physical-ai-natnew.md](../../sources/repos/awesome-physical-ai-natnew.md)",
        "- [sources/repos/awesome-physical-ai-aichr.md](../../sources/repos/awesome-physical-ai-aichr.md)",
    ]
    return f"""---
type: entity
tags: [{", ".join(tags)}]
status: complete
updated: {TODAY}
{fm_extra_s}summary: "{summary}"
related:
{_yaml_list(related)}
sources:
  - ../../{src_rel}
  - ../../{CATALOG}
  - ../../sources/repos/awesome-physical-ai-natnew.md
  - ../../sources/repos/awesome-physical-ai-aichr.md
---

# {short}

**{e["title"]}** 收录于 awesome-physical-ai（{lists}）**第 {idx:03d}/{total:03d}** 条，分组 **{sections}**。本页为知识库 **策展索引级** 详情节点；细节以官方文档 / 原文为准。

## 一句话定义

{e["desc"]}

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
{abbrev_rows}

## 为什么重要

- {e["desc"]}
- 在 [Physical AI 技术地图](../overview/awesome-physical-ai-technology-map.md) 中提供可点击的独立详情节点，避免清单条目无法落入知识图谱。
- 双清单去重后只保留一个 canonical 节点，并同时引用 natnew / aichr 来源。

## 核心信息（索引级）

| 字段 | 内容 |
|------|------|
| 编号 | {idx:03d}/{total:03d} |
| 分组 | {sections} |
| 来源清单 | {lists} |
| 主链接 | <{primary}> |{arxiv_row}{gh_row}
{paper_sections}
## 局限与风险

- 不要把 Awesome 摘要当成完整方法证明或合规结论。
- 同名 GitHub 仓（natnew vs aichr）条目链接可能不同；以本页主链接与技术地图为准。
- 清单中的实验室 / 硬件 / 人物条目偶发链到错误 org，复现或引用前先打开官方页核对。

## 关联页面

- [awesome-physical-ai（natnew）](../entities/awesome-physical-ai-natnew.md)
- [awesome-physical-ai（aichr）](../entities/awesome-physical-ai-aichr.md)
- [Physical AI 技术地图](../overview/awesome-physical-ai-technology-map.md)
- [Physical AI 策展清单对比](../comparisons/awesome-physical-ai-curated-lists.md)

## 参考来源

{chr(10).join(src_links)}
- 主链接：<{primary}>

## 推荐继续阅读

- [natnew/awesome-physical-ai]({NATNEW_URL})
- [aichr/awesome-physical-ai]({AICHR_URL})
- [原文 / 官方入口]({primary})
"""


def render_catalog(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# awesome-physical-ai 并集目录（natnew ∪ aichr）",
        "",
        f"> 由 `{NATNEW_URL}` 与 `{AICHR_URL}` 解析生成；入库日 {TODAY}。",
        "",
        f"- **技术地图：** [`{TECH_MAP}`](../../{TECH_MAP})",
        f"- **列表实体：** [natnew](../../{NATNEW_ENTITY}) · [aichr](../../{AICHR_ENTITY})",
        f"- **去重后条目：** {len(rows)}",
        "",
        "| # | 标题 | ID | 分组 | 清单 | wiki | 状态 |",
        "|---|------|----|------|------|------|------|",
    ]
    for r in rows:
        sid = r["sid"]
        wiki = r["wiki_rel"]
        status = "新建" if r.get("created") else "复用"
        lists = "+".join(r["lists"])
        sec = r["sections"][0][:28]
        lines.append(
            f"| {r['idx']:03d} | {r['title'][:70]} | `{sid[:40]}` | {sec} | {lists} | "
            f"[`{Path(wiki).name}`](../../{wiki}) | {status} |"
        )
    lines.append("")
    return "\n".join(lines)


def render_tech_map(rows: list[dict[str, Any]]) -> str:
    by_sec: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_sec[r["sections"][0]].append(r)
    new_count = sum(1 for r in rows if r.get("created"))
    reused = len(rows) - new_count
    sections_md: list[str] = []
    for sec, items in by_sec.items():
        sections_md.append(f"### {sec}\n")
        sections_md.append("| # | 条目 | 详情节点 | 来源 |")
        sections_md.append("|---|------|----------|------|")
        for r in items:
            link = wiki_rel_from_root(r["wiki_rel"])
            title = r["title"].replace("|", "/")
            lists = "+".join(r["lists"])
            sections_md.append(
                f"| {r['idx']:03d} | {title[:80]} | [{Path(r['wiki_rel']).stem}]({link}) | {lists} |"
            )
        sections_md.append("")
    return f"""---
type: overview
tags: [overview, curated-index, physical-ai, awesome-physical-ai, technology-map]
status: complete
updated: {TODAY}
summary: "Physical AI 双清单技术地图：natnew ∪ aichr 去重后 {len(rows)} 条，新建 {new_count}、复用 {reused}。"
related:
  - ../entities/awesome-physical-ai-natnew.md
  - ../entities/awesome-physical-ai-aichr.md
  - ../comparisons/awesome-physical-ai-curated-lists.md
  - ../methods/vla.md
  - ../concepts/sim2real.md
sources:
  - ../../{CATALOG}
  - ../../sources/repos/awesome-physical-ai-natnew.md
  - ../../sources/repos/awesome-physical-ai-aichr.md
---

# Awesome Physical AI 技术地图

> 本页把 [natnew/awesome-physical-ai]({NATNEW_URL}) 与 [aichr/awesome-physical-ai]({AICHR_URL}) 的清单条目映射为站内 **独立详情节点**（新建 `pai-*` / `paper-pai-*` 或复用已有 canonical 页）。

## 一句话定义

**Physical AI 双清单技术地图** = 两份同名 Awesome 列表的并集节点化索引（按清单分组浏览，一点即达详情页）。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| PAI | Physical AI | 感知–推理–行动闭环的物理智能 |
| VLA | Vision-Language-Action | 两清单共同主线 |
| RFM | Robotics Foundation Model | natnew canonical 类 |
| Sim2Real | Simulation to Real | 迁移与评测独立类 |

## 为什么重要

- Awesome 列表本身不是知识图谱节点；若不升格子条目，图谱只能停在清单 hub。
- 本地图 **优先复用** 库内已有 arXiv / GitHub / 标题 canonical 页，仅对缺失条目新建索引级节点。
- 统计：去重后 **{len(rows)}** 条（新建 **{new_count}**，复用 **{reused}**）。

## 覆盖范围

| 项 | 值 |
|----|-----|
| 上游 | <{NATNEW_URL}> · <{AICHR_URL}> |
| 目录 source | [awesome-physical-ai-union-catalog.md](../../{CATALOG}) |
| 对照页 | [natnew vs aichr](../comparisons/awesome-physical-ai-curated-lists.md) |

## 分组索引

{chr(10).join(sections_md)}

## 局限与风险

- 索引级节点保留清单摘要，**不替代** 深度论文/工具页。
- 人物、新闻通讯与部分实验室条目只有社交媒体或新闻链接，节点用于图谱覆盖而非复现。
- 上游更新后需重跑 `python3 scripts/generate_physical_ai_awesome_entities.py` 再 `make ci-preflight`。

## 关联页面

- [awesome-physical-ai（natnew）](../entities/awesome-physical-ai-natnew.md)
- [awesome-physical-ai（aichr）](../entities/awesome-physical-ai-aichr.md)
- [Physical AI 策展清单对比](../comparisons/awesome-physical-ai-curated-lists.md)
- [VLA](../methods/vla.md)
- [Sim2Real](../concepts/sim2real.md)

## 参考来源

- [awesome-physical-ai-union-catalog.md](../../{CATALOG})
- [sources/repos/awesome-physical-ai-natnew.md](../../sources/repos/awesome-physical-ai-natnew.md)
- [sources/repos/awesome-physical-ai-aichr.md](../../sources/repos/awesome-physical-ai-aichr.md)

## 推荐继续阅读

- [natnew GitHub]({NATNEW_URL})
- [aichr GitHub]({AICHR_URL})
- [natnew docs overview](https://natnew.github.io/awesome-physical-ai/docs/overview)
"""


def update_hubs(rows: list[dict[str, Any]]) -> None:
    new_count = sum(1 for r in rows if r.get("created"))
    reused = len(rows) - new_count
    both = sum(1 for r in rows if len(r["lists"]) > 1)
    block = f"""
## 子节点覆盖（2026-09-20 纵深）

去重后 **{len(rows)}** 条独立详情节点（新建 {new_count}，复用 {reused}；两清单同时出现 {both}）。

完整子节点表见 [Physical AI 技术地图](../overview/awesome-physical-ai-technology-map.md)；并集目录见 [awesome-physical-ai-union-catalog.md](../../sources/repos/awesome-physical-ai-union-catalog.md)。

"""
    natnew = ROOT / NATNEW_ENTITY
    text = natnew.read_text(encoding="utf-8")
    text = text.replace("updated: 2026-09-19", f"updated: {TODAY}")
    if "../overview/awesome-physical-ai-technology-map.md" not in text:
        text = text.replace(
            "  - ./lerobot.md\n",
            "  - ../overview/awesome-physical-ai-technology-map.md\n  - ./lerobot.md\n",
            1,
        )
    if "子节点覆盖" not in text:
        text = text.replace("## 核心结构", block + "## 核心结构", 1)
    natnew.write_text(text, encoding="utf-8")

    aichr = ROOT / AICHR_ENTITY
    text = aichr.read_text(encoding="utf-8")
    text = text.replace("updated: 2026-09-19", f"updated: {TODAY}")
    if "../overview/awesome-physical-ai-technology-map.md" not in text:
        text = text.replace(
            "  - ./lerobot.md\n",
            "  - ../overview/awesome-physical-ai-technology-map.md\n  - ./lerobot.md\n",
            1,
        )
    if "子节点覆盖" not in text:
        text = text.replace("## 核心结构", block + "## 核心结构", 1)
    aichr.write_text(text, encoding="utf-8")

    cmp_p = ROOT / COMPARE_ENTITY
    text = cmp_p.read_text(encoding="utf-8")
    text = text.replace("updated: 2026-09-19", f"updated: {TODAY}")
    if "../overview/awesome-physical-ai-technology-map.md" not in text:
        text = text.replace(
            "  - ../queries/embodied-fm-taxonomy-loop.md\n",
            "  - ../overview/awesome-physical-ai-technology-map.md\n"
            "  - ../queries/embodied-fm-taxonomy-loop.md\n",
            1,
        )
    if "子节点覆盖" not in text:
        text = text.replace("## 怎么选", block + "## 怎么选", 1)
    cmp_p.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    natnew = CACHE_DIR / "natnew-README.md"
    aichr = CACHE_DIR / "aichr-README.md"
    if not natnew.exists() or not aichr.exists():
        raise SystemExit(f"missing cached READMEs in {CACHE_DIR}")

    raw = parse_readme(natnew, "natnew") + parse_readme(aichr, "aichr")
    items = union_entries(raw)
    index = WikiIndex()
    index.load()

    rows: list[dict[str, Any]] = []
    created_entities = 0
    created_sources = 0
    reused = 0
    ambiguous: list[str] = []
    used_names: set[str] = set()

    for i, e in enumerate(items, start=1):
        existing = resolve_existing(e, index)
        created = False
        src_rel = CATALOG
        if existing:
            wiki_rel = existing
            reused += 1
        else:
            wiki_rel = new_wiki_rel(e, i)
            # collision guard
            base = wiki_rel
            n = 2
            while wiki_rel in used_names or (ROOT / wiki_rel).exists():
                wiki_rel = base.replace(".md", f"-{n}.md")
                n += 1
            src_rel = new_source_rel(e, i)
            if not args.dry_run:
                src_path = ROOT / src_rel
                ent_path = ROOT / wiki_rel
                src_path.parent.mkdir(parents=True, exist_ok=True)
                if not src_path.exists():
                    src_path.write_text(render_source(e, wiki_rel, i, len(items)), encoding="utf-8")
                    created_sources += 1
                if not ent_path.exists():
                    ent_path.write_text(render_entity(e, src_rel, i, len(items)), encoding="utf-8")
                    created_entities += 1
                    created = True
                else:
                    reused += 1
            else:
                created_entities += 1
                created = True
        if e.get("arxiv"):
            index.arxiv.setdefault(e["arxiv"], wiki_rel)
        if e.get("github"):
            index.github.setdefault(e["github"], wiki_rel)
        used_names.add(wiki_rel)
        if "apollo.auto" in (e.get("url") or "") or "rise-lab-skku" in (e.get("url") or ""):
            ambiguous.append(f"{e['title']} → {e.get('url')}")
        rows.append({**e, "idx": i, "wiki_rel": wiki_rel, "created": created, "src_rel": src_rel})

    stats = {
        "raw_natnew": sum(1 for x in raw if x["list"] == "natnew"),
        "raw_aichr": sum(1 for x in raw if x["list"] == "aichr"),
        "union": len(items),
        "created_entities": created_entities,
        "created_sources": created_sources,
        "reused": reused,
        "both_lists": sum(1 for r in rows if len(r["lists"]) > 1),
        "ambiguous": ambiguous,
        "by_kind": {},
    }
    kind_counts: dict[str, int] = defaultdict(int)
    for r in rows:
        kind_counts[r["kind"]] += 1
    stats["by_kind"] = dict(kind_counts)

    if not args.dry_run:
        (ROOT / CATALOG).write_text(render_catalog(rows), encoding="utf-8")
        (ROOT / TECH_MAP).write_text(render_tech_map(rows), encoding="utf-8")
        update_hubs(rows)
        for src_name, list_key in (
            ("awesome-physical-ai-natnew.md", "natnew"),
            ("awesome-physical-ai-aichr.md", "aichr"),
        ):
            p = ROOT / "sources" / "repos" / src_name
            text = p.read_text(encoding="utf-8")
            if "union-catalog" not in text:
                text += (
                    f"\n## 纵深节点化（{TODAY}）\n\n"
                    f"- 并集目录：[`{CATALOG}`](../../{CATALOG})\n"
                    f"- 技术地图：[`{TECH_MAP}`](../../{TECH_MAP})\n"
                    f"- 本清单解析条目："
                    f"{sum(1 for r in rows if list_key in r['lists'])}\n"
                )
                p.write_text(text, encoding="utf-8")

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    STATS_PATH.write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(stats, indent=2, ensure_ascii=False))
    if args.dry_run:
        missing = [r for r in rows if r.get("created")]
        print(f"dry-run would create {len(missing)} pages; sample:")
        for r in missing[:40]:
            print(f"  NEW {r['kind']:10} {r['title'][:60]} -> {r['wiki_rel']}")


if __name__ == "__main__":
    main()
