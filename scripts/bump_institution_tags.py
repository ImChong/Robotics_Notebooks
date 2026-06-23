#!/usr/bin/env python3
"""为 wiki 页面补全 frontmatter tags 中的机构别名，以派生「所属机构」。

规则（与 generate_link_graph.derive_node_institutions 对齐）：
  1. 已有 institutions 派生结果则跳过；
  2. 显式 frontmatter `institutions:` 非空则跳过；
  3. 从 tags / 摘要区 / H1 首段 / 显式覆盖表推断机构，将 **alias token** 写入 tags。

用法：
  python3 scripts/bump_institution_tags.py --dry-run
  python3 scripts/bump_institution_tags.py
  python3 scripts/bump_institution_tags.py --tools-only
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
WIKI_DIR = REPO_ROOT / "wiki"
INSTITUTIONS_PATH = REPO_ROOT / "schema" / "institutions.json"

TOOL_TAGS = frozenset(
    {
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
    }
)

# 纯硬件/平台概念页（非软件工具）
HARDWARE_PLATFORM_TAGS = frozenset(
    {"hardware", "platform", "humanoid", "quadruped", "legged", "actuator", "locomotion"}
)

TOOL_NAME_HINTS = ("-sim", "_sim", "-gym", "-lab", "-ros", "-sdk", "-lib", "-slam", "-vio", "-vslam")

# 短 alias 在正文易误命中，仅允许出现在 tags 或显式覆盖。
SHORT_ALIASES = frozenset({"eth", "mit", "meta", "hku", "pku", "zju", "sfu", "tri"})

# wiki 相对路径 → canonical institution id（覆盖表，优先于正文推断）
PAGE_INSTITUTION_OVERRIDES: dict[str, list[str]] = {
    "wiki/entities/isaac-gym.md": ["nvidia"],
    "wiki/entities/isaac-lab.md": ["nvidia"],
    "wiki/entities/isaac-gym-isaac-lab.md": ["nvidia"],
    "wiki/entities/legged-gym.md": ["eth"],
    "wiki/entities/lerobot.md": ["huggingface"],
    "wiki/entities/openvla.md": ["stanford"],
    "wiki/entities/ai2-thor.md": ["ai2"],
    "wiki/entities/holosoma.md": ["amazon"],
    "wiki/entities/pinocchio.md": ["inria"],
    "wiki/entities/drake.md": ["toyota-research", "mit"],
    "wiki/entities/maniskill2.md": ["ucsd"],
    "wiki/entities/sapien.md": ["ucsd"],
    "wiki/entities/mimickit.md": ["nvidia"],
    "wiki/entities/newton-physics.md": ["linux-foundation", "nvidia"],
    "wiki/entities/metahuman.md": ["epic-games"],
    "wiki/entities/airsim.md": ["microsoft"],
    "wiki/entities/crazyflie-firmware.md": ["bitcraze"],
    "wiki/entities/crazyswarm2.md": ["bitcraze"],
    "wiki/entities/holomotion.md": ["horizon-robotics"],
    "wiki/entities/genesis-sim.md": ["genesis-ai"],
    "wiki/entities/genesis-world-10.md": ["genesis-ai"],
    "wiki/entities/gene-26-5-genesis-ai.md": ["genesis-ai"],
    "wiki/entities/openloong-dyn-control.md": ["openloong"],
    "wiki/entities/amp-rsl-rl.md": ["eth"],
    "wiki/entities/amp-for-hardware.md": ["nvidia"],
    "wiki/entities/human2humanoid.md": ["cmu"],
    "wiki/entities/ego-planner-swarm.md": ["zju"],
    "wiki/entities/paper-mighty-hermite-spline-trajectory-planning.md": ["mit"],
    "wiki/entities/sbto.md": ["mit"],
    "wiki/entities/axellwppr-motion-tracking.md": ["unitree"],
    "wiki/entities/mjlab.md": ["nvidia"],
    "wiki/entities/humanoid-gym.md": ["nvidia"],
    "wiki/entities/metalhead.md": ["unitree", "eth", "nvidia"],
    "wiki/entities/leggedgym-ex.md": ["eth"],
    "wiki/entities/aloha.md": ["google-deepmind", "stanford"],
    "wiki/entities/carla.md": ["microsoft"],
    "wiki/entities/behavior-1k.md": ["nvidia"],
    "wiki/entities/robogen.md": ["stanford"],
    "wiki/entities/matterport3d-simulator.md": ["princeton"],
    "wiki/entities/flightmare.md": ["eth"],
    "wiki/entities/fairmotion.md": ["meta"],
    "wiki/entities/rldx-1.md": ["huggingface"],
    "wiki/entities/robot-io-rio.md": ["nvidia"],
    "wiki/entities/unilab.md": ["tsinghua"],
    "wiki/entities/pytorch.md": ["meta"],
    "wiki/entities/blender.md": ["blender-foundation"],
    "wiki/entities/paper-heracles-humanoid-diffusion.md": ["x-humanoid"],
    "wiki/entities/jackhan-mujoco-walke3-simulation.md": ["sdu"],
    "wiki/entities/spear-sim.md": ["nvidia"],
    "wiki/entities/gs-playground.md": ["nvidia"],
    "wiki/entities/go2-motion-imitation.md": ["unitree"],
    "wiki/entities/wbc-fsm.md": ["unitree"],
    "wiki/entities/stmr-quadruped-retargeting.md": ["eth"],
    "wiki/entities/motion-imitation-quadruped.md": ["berkeley"],
    "wiki/entities/phc.md": ["nvidia"],
    "wiki/entities/videomimic.md": ["nvidia"],
    "wiki/entities/gen2humanoid.md": ["baai"],
    "wiki/entities/gvhmr.md": ["tsinghua"],
    "wiki/entities/gym-pybullet-drones.md": ["mit"],
    "wiki/entities/quad-swarm-rl.md": ["mit"],
    "wiki/entities/mushr.md": ["mit"],
    "wiki/entities/python-robotics.md": ["mit"],
    "wiki/entities/open-vins.md": ["cmu"],
    "wiki/entities/orb-slam3.md": ["eth"],
    "wiki/entities/openvslam.md": ["eth"],
    "wiki/entities/vins-fusion.md": ["hku"],
    "wiki/entities/fast-lio.md": ["hku"],
    "wiki/entities/lio-sam.md": ["hku"],
    "wiki/entities/hdl-graph-slam.md": ["hku"],
    "wiki/entities/lego-loam.md": ["hku"],
    "wiki/entities/rtabmap.md": ["hku"],
    "wiki/entities/cartographer.md": ["google"],
    "wiki/entities/navigation2.md": ["linux-foundation"],
    "wiki/entities/slam-toolbox.md": ["linux-foundation"],
    "wiki/entities/plotjuggler.md": ["linux-foundation"],
    "wiki/entities/autoware.md": ["linux-foundation"],
    "wiki/entities/px4-autopilot.md": ["linux-foundation"],
    "wiki/entities/mavsdk.md": ["linux-foundation"],
    "wiki/entities/xtdrone.md": ["linux-foundation"],
    "wiki/entities/betaflight.md": ["linux-foundation"],
    "wiki/entities/simplefoc.md": ["linux-foundation"],
    "wiki/entities/wtfos.md": ["linux-foundation"],
    "wiki/entities/matrix-simulation-platform.md": ["nvidia"],
    "wiki/entities/mobilegym.md": ["tsinghua"],
    "wiki/entities/rf-detr.md": ["nvidia"],
    "wiki/entities/ppf-contact-solver.md": ["nvidia"],
    "wiki/entities/april-tag.md": ["mit"],
    "wiki/entities/freemocap.md": ["freemocap"],
    "wiki/entities/manim.md": [],
    "wiki/entities/spark-3dgs-renderer.md": ["linux-foundation"],
    "wiki/entities/aholo-viewer.md": ["linux-foundation"],
    "wiki/entities/bam-better-actuator-models.md": ["google"],
    "wiki/entities/botworld.md": ["nvidia"],
    "wiki/entities/walk-the-dog.md": ["nvidia"],
    "wiki/entities/pan-motion-retargeting.md": ["nvidia"],
    "wiki/entities/mocap-retarget.md": ["nvidia"],
    "wiki/entities/robot-motion-keyframe-editors.md": ["nvidia"],
    "wiki/entities/tnkr.md": ["linux-foundation"],
    "wiki/entities/world-labs.md": ["nvidia"],
    "wiki/entities/asimov-v1.md": ["unitree"],
    "wiki/entities/atom01-deploy.md": ["linux-foundation"],
    "wiki/entities/awesome-text-to-motion-zilize.md": ["linux-foundation"],
    "wiki/entities/jackhan-mujoco-walke3-simulation.md": [],
    "wiki/entities/paper-slowrl-safe-lora-locomotion-sim2real.md": ["unitree", "nvidia"],
    "wiki/entities/paper-quadruped-agile-sim2real-rss2018.md": ["mit"],
    "wiki/entities/paper-cassie-iterative-locomotion-sim2real.md": ["berkeley"],
    "wiki/entities/paper-hrl-stack-29-opening_the_sim_to_real_door_for_hum.md": ["nvidia"],
}


def _load_registry(path: Path) -> dict[str, dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    registry = data.get("registry", {})
    return registry if isinstance(registry, dict) else {}


def _build_alias_map(registry: dict[str, dict]) -> dict[str, str]:
    alias_map: dict[str, str] = {}
    for canonical_id, meta in registry.items():
        alias_map[canonical_id.lower()] = canonical_id
        for alias in meta.get("aliases", []) or []:
            alias_map[str(alias).strip().lower()] = canonical_id
    return alias_map


def _preferred_alias(canonical_id: str, registry: dict[str, dict]) -> str:
    aliases = registry.get(canonical_id, {}).get("aliases", []) or []
    if aliases:
        return str(aliases[0]).strip().lower()
    return canonical_id.lower()


def _parse_frontmatter_list(content: str, key: str) -> list[str]:
    if not content.startswith("---"):
        return []
    end = content.find("\n---", 3)
    if end == -1:
        return []
    fm = content[3:end]
    inline = re.search(rf"^{re.escape(key)}\s*:\s*\[(.*?)\]", fm, re.MULTILINE)
    if inline:
        return [item.strip().strip("'\"") for item in inline.group(1).split(",") if item.strip()]
    block = re.search(rf"^{re.escape(key)}\s*:\s*\n((?:[ \t]*-[ \t]*.+\n?)+)", fm, re.MULTILINE)
    if block:
        items = re.findall(r"-[ \t]*(.+)", block.group(1))
        return [item.strip().strip("'\"") for item in items if item.strip()]
    return []


def _derive_institutions(content: str, alias_map: dict[str, str]) -> list[str]:
    explicit = _parse_frontmatter_list(content, "institutions")
    source = explicit if explicit else _parse_frontmatter_list(content, "tags")
    out: list[str] = []
    seen: set[str] = set()
    for token in source:
        canonical = alias_map.get(str(token).strip().lower())
        if canonical and canonical not in seen:
            seen.add(canonical)
            out.append(canonical)
    return out


def _scan_region(content: str) -> str:
    """摘要区 + H1 + 首段（至第一个 ##），避免 related/参考来源 里的交叉引用误命中。"""
    parts: list[str] = []
    if content.startswith("---"):
        end = content.find("\n---", 3)
        if end != -1:
            fm = content[3:end]
            for key in ("summary", "title", "description"):
                m = re.search(rf"^{key}:\s*(.+)$", fm, re.MULTILINE)
                if m:
                    parts.append(m.group(1).strip().strip("'\""))
    body = content.split("\n---\n", 2)[-1] if content.startswith("---") else content
    h1 = re.search(r"^#\s+.+$", body, re.MULTILINE)
    if h1:
        parts.append(h1.group(0))
    after_h1 = body[h1.end() :] if h1 else body
    section_end = after_h1.find("\n## ")
    intro = after_h1[:section_end] if section_end != -1 else after_h1[:1200]
    parts.append(intro)
    return "\n".join(parts).lower()


def _label_keywords(registry: dict[str, dict]) -> list[tuple[str, str]]:
    """(keyword_lower, canonical_id) 按关键词长度降序，优先长匹配。"""
    pairs: list[tuple[str, str]] = []
    for cid, meta in registry.items():
        label = str(meta.get("label", ""))
        if label:
            pairs.append((label.lower(), cid))
        paren = re.search(r"（([^）]+)）", label)
        if paren:
            pairs.append((paren.group(1).lower(), cid))
        for alias in meta.get("aliases", []) or []:
            al = str(alias).strip().lower()
            if len(al) >= 4:
                pairs.append((al, cid))
    pairs.sort(key=lambda x: -len(x[0]))
    return pairs


def is_tool_entity(rel_path: str, tags: list[str]) -> bool:
    parts = Path(rel_path).parts
    if len(parts) < 2 or parts[0] != "wiki" or parts[1] != "entities":
        return False
    stem = Path(rel_path).stem
    tag_set = {t.lower() for t in tags}
    if "paper-notebook" in stem or "paper-notebook-stub" in tag_set or "paper-notebook-planned" in tag_set:
        return False
    if tag_set <= {"paper", "humanoid-paper-notebooks"} or "paper-notebook" in tag_set:
        return False
    if stem.startswith("paper-"):
        return bool(tag_set & TOOL_TAGS)
    if tag_set <= HARDWARE_PLATFORM_TAGS or (
        tag_set & {"hardware", "platform"} and not tag_set & TOOL_TAGS
    ):
        return False
    if tag_set & TOOL_TAGS:
        return True
    return any(hint in stem for hint in TOOL_NAME_HINTS)


def _keyword_in_text(keyword: str, text: str) -> bool:
    kw = keyword.lower()
    if re.search(r"[\u4e00-\u9fff]", kw):
        return kw in text
    if kw == "mit":
        if re.search(r"\bmit\s+license\b", text):
            return False
    return bool(re.search(r"(?<![a-z0-9])" + re.escape(kw) + r"(?![a-z0-9])", text))


def infer_institution_ids(
    rel_path: str,
    content: str,
    registry: dict[str, dict],
    alias_map: dict[str, str],
    label_keywords: list[tuple[str, str]],
) -> list[str]:
    if _derive_institutions(content, alias_map):
        return []

    found: list[str] = []
    seen: set[str] = set()

    def add(cid: str) -> None:
        if cid not in seen:
            seen.add(cid)
            found.append(cid)

    for cid in PAGE_INSTITUTION_OVERRIDES.get(rel_path, []):
        add(cid)

    tags = _parse_frontmatter_list(content, "tags")
    for token in tags:
        cid = alias_map.get(token.lower())
        if cid:
            add(cid)

    scan = _scan_region(content)
    for keyword, cid in label_keywords:
        if _keyword_in_text(keyword, scan):
            add(cid)

    for alias, cid in alias_map.items():
        if alias in SHORT_ALIASES:
            continue
        if len(alias) < 4:
            continue
        if _keyword_in_text(alias, scan):
            add(cid)

    for alias in SHORT_ALIASES:
        cid = alias_map.get(alias)
        if not cid or cid in seen:
            continue
        if alias in {t.lower() for t in tags}:
            add(cid)

    return found


def _add_tags_to_frontmatter(content: str, new_tags: list[str]) -> str:
    if not content.startswith("---"):
        return content
    end = content.find("\n---", 3)
    if end == -1:
        return content
    fm = content[3:end]
    rest = content[end:]

    existing = _parse_frontmatter_list(content, "tags")
    existing_lower = {t.lower() for t in existing}
    to_add = [t for t in new_tags if t.lower() not in existing_lower]
    if not to_add:
        return content
    merged = existing + to_add

    inline = re.search(r"^tags\s*:\s*\[(.*?)\]", fm, re.MULTILINE)
    if inline:
        new_fm = re.sub(
            r"^tags\s*:\s*\[.*?\]",
            "tags: [" + ", ".join(merged) + "]",
            fm,
            count=1,
            flags=re.MULTILINE,
        )
    else:
        new_fm = fm.rstrip() + "\ntags: [" + ", ".join(merged) + "]\n"

    return f"---\n{new_fm}{rest}"


def bump_page(
    path: Path,
    registry: dict[str, dict],
    alias_map: dict[str, str],
    label_keywords: list[tuple[str, str]],
    *,
    dry_run: bool,
) -> list[str] | None:
    rel = path.relative_to(REPO_ROOT).as_posix()
    content = path.read_text(encoding="utf-8")
    if _derive_institutions(content, alias_map):
        return None

    cids = infer_institution_ids(rel, content, registry, alias_map, label_keywords)
    if not cids:
        return None

    new_tags = [_preferred_alias(cid, registry) for cid in cids]
    if dry_run:
        return new_tags

    new_content = _add_tags_to_frontmatter(content, new_tags)
    if new_content != content:
        path.write_text(new_content, encoding="utf-8")
    return new_tags


def main() -> int:
    parser = argparse.ArgumentParser(description="Bump institution alias tags on wiki pages")
    parser.add_argument("--dry-run", action="store_true", help="只打印将修改的页面")
    parser.add_argument("--tools-only", action="store_true", help="仅处理 entities 工具页")
    args = parser.parse_args()

    registry = _load_registry(INSTITUTIONS_PATH)
    alias_map = _build_alias_map(registry)
    label_keywords = _label_keywords(registry)

    changed: list[tuple[str, list[str]]] = []
    tool_still_missing: list[str] = []

    for path in sorted(WIKI_DIR.rglob("*.md")):
        rel = path.relative_to(REPO_ROOT).as_posix()
        tags = _parse_frontmatter_list(path.read_text(encoding="utf-8"), "tags")
        if args.tools_only and not is_tool_entity(rel, tags):
            continue

        added = bump_page(path, registry, alias_map, label_keywords, dry_run=args.dry_run)
        if added:
            changed.append((rel, added))

    if not args.dry_run:
        for path in sorted(WIKI_DIR.rglob("*.md")):
            rel = path.relative_to(REPO_ROOT).as_posix()
            if not is_tool_entity(rel, _parse_frontmatter_list(path.read_text(encoding="utf-8"), "tags")):
                continue
            content = path.read_text(encoding="utf-8")
            if not _derive_institutions(content, alias_map):
                tool_still_missing.append(rel)

    print(f"{'[dry-run] ' if args.dry_run else ''}updated {len(changed)} pages")
    for rel, tags in changed[:80]:
        print(f"  {rel}: +{tags}")
    if len(changed) > 80:
        print(f"  ... and {len(changed) - 80} more")

    if tool_still_missing:
        print(f"\nTool entities still missing institutions ({len(tool_still_missing)}):")
        for rel in tool_still_missing[:30]:
            print(f"  {rel}")
        if len(tool_still_missing) > 30:
            print(f"  ... and {len(tool_still_missing) - 30} more")

    return 0


if __name__ == "__main__":
    sys.exit(main())
