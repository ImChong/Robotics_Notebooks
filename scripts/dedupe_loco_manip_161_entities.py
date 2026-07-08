#!/usr/bin/env python3
"""Merge duplicate loco-manip-161 survey stubs into canonical wiki entities.

Deletes `wiki/entities/paper-loco-manip-161-*` stubs when the same work already
has a deeper canonical page (hrl-stack / bfm / amp / loco-manip-8 / methods / …).
Updates catalog, category hubs, survey sources, and bootstrap mapping.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENTITIES = ROOT / "wiki" / "entities"
METHODS = ROOT / "wiki" / "methods"
BOOTSTRAP = ROOT / "scripts" / "bootstrap_loco_manip_161_entities.py"
CATALOG = ROOT / "sources" / "papers" / "humanoid_loco_manip_161_catalog.md"

# stub basename -> (wiki subdir, canonical basename without .md)
# Excludes false links (e.g. REFINE-DP ≠ monocular HMR) and intra-161-only keepers.
MERGE_MAP: dict[str, tuple[str, str]] = {
    # cross-survey via 同题深读 (entities)
    "paper-loco-manip-161-001-agility-meets-stability": ("methods", "ams"),
    "paper-loco-manip-161-002-any2any": ("entities", "paper-any2any-cross-embodiment-wbt"),
    "paper-loco-manip-161-003-bfm-zero": ("entities", "paper-bfm-zero"),
    "paper-loco-manip-161-004-beyondmimic": ("entities", "paper-beyondmimic"),
    "paper-loco-manip-161-005-chip": ("entities", "paper-hrl-stack-36-chip"),
    "paper-loco-manip-161-006-clone": ("entities", "paper-bfm-12-clone"),
    "paper-loco-manip-161-010-hover": ("entities", "paper-bfm-14-hover"),
    "paper-loco-manip-161-011-holomotion": ("entities", "holomotion"),
    "paper-loco-manip-161-013-kungfubot2": ("entities", "paper-notebook-kungfubot-2"),
    "paper-loco-manip-161-015-make-tracking-easy": (
        "entities",
        "paper-hrl-stack-02-make_tracking_easy",
    ),
    "paper-loco-manip-161-016-omnih2o": ("entities", "paper-hrl-stack-08-omnih2o"),
    "paper-loco-manip-161-017-omniretarget": ("entities", "paper-hrl-stack-03-omniretarget"),
    "paper-loco-manip-161-018-retargeting": ("entities", "paper-hrl-stack-01-retargeting_matters"),
    "paper-loco-manip-161-020-twist2": ("entities", "paper-twist2"),
    "paper-loco-manip-161-021-twist": ("entities", "paper-twist"),
    "paper-loco-manip-161-033-ceer": ("entities", "paper-motion-cerebellum-ceer"),
    "paper-loco-manip-161-034-cola": (
        "entities",
        "paper-notebook-learning-human-humanoid-coordination-for-collabo",
    ),
    "paper-loco-manip-161-035-coordex": (
        "entities",
        "paper-coordex-dexterous-humanoid-loco-manipulation",
    ),
    "paper-loco-manip-161-037-handoff": ("entities", "paper-motion-cerebellum-handoff"),
    "paper-loco-manip-161-043-pilot": ("entities", "paper-pilot-perceptive-loco-manipulation"),
    "paper-loco-manip-161-044-rpl": ("entities", "paper-rpl-robust-humanoid-perceptive-locomotion"),
    "paper-loco-manip-161-045-splitadapter": (
        "entities",
        "paper-splitadapter-load-aware-loco-manipulation",
    ),
    "paper-loco-manip-161-051-wholebodyvla": ("entities", "paper-hrl-stack-30-wholebodyvla"),
    "paper-loco-manip-161-063-halomi": ("entities", "paper-halomi-humanoid-loco-manipulation"),
    "paper-loco-manip-161-070-oasis": ("entities", "paper-loco-manip-04-oasis"),
    "paper-loco-manip-161-073-physhsi": ("entities", "paper-amp-survey-15-physhsi"),
    "paper-loco-manip-161-082-visualmimic": ("entities", "paper-notebook-visualmimic"),
    "paper-loco-manip-161-083-wholebodyvla": ("entities", "paper-hrl-stack-30-wholebodyvla"),
    "paper-loco-manip-161-095-egoprimo": ("entities", "paper-loco-manip-02-egoprimo"),
    "paper-loco-manip-161-098-langwbc": ("entities", "paper-bfm-37-langwbc"),
    "paper-loco-manip-161-099-motiondisco": (
        "entities",
        "paper-motiondisco-extreme-humanoid-loco-manipulation",
    ),
    "paper-loco-manip-161-100-motionwam": (
        "entities",
        "paper-motionwam-humanoid-loco-manipulation-wam",
    ),
    "paper-loco-manip-161-101-omg": ("entities", "paper-omg-omni-modal-humanoid-control"),
    "paper-loco-manip-161-112-humanx": ("entities", "paper-hrl-stack-05-humanx"),
    "paper-loco-manip-161-114-omniretarget": ("entities", "paper-hrl-stack-03-omniretarget"),
    "paper-loco-manip-161-115-resmimic": ("entities", "paper-resmimic"),
    "paper-loco-manip-161-120-gentlehumanoid": ("entities", "paper-hrl-stack-37-gentlehumanoid"),
    "paper-loco-manip-161-122-leverb": ("entities", "paper-bfm-36-leverb"),
    "paper-loco-manip-161-128-bifrostumi": ("entities", "paper-bifrost-umi"),
    "paper-loco-manip-161-132-twist2": ("entities", "paper-twist2"),
    "paper-loco-manip-161-133-twist": ("entities", "paper-twist"),
    "paper-loco-manip-161-140-twist2": ("entities", "paper-twist2"),
    "paper-loco-manip-161-146-agibot-world-colosseo": ("entities", "agibot-world-2026"),
    "paper-loco-manip-161-149-gemini-robotics": ("entities", "gemini-robotics"),
    "paper-loco-manip-161-150-genie-envisioner": ("entities", "ge-sim-2"),
    "paper-loco-manip-161-151-legs": ("entities", "paper-legs-embodied-gaussian-splatting-vla"),
    "paper-loco-manip-161-153-metaworld-x": ("entities", "paper-hrl-stack-32-metaworld"),
    "paper-loco-manip-161-158-rove": ("entities", "paper-rove-humanoid-vla-intervention"),
    # cross-survey via 同题深读 (methods)
    "paper-loco-manip-161-046-sumo": ("methods", "sumo"),
    "paper-loco-manip-161-147-dial": ("methods", "dial-instruction-augmentation"),
    "paper-loco-manip-161-152-lingbot-vla": ("entities", "lingbot-vla-v2"),
    # intra-161 duplicate slots → keeper stub (still loco-manip entity)
    "paper-loco-manip-161-008-from-w1": ("entities", "paper-loco-manip-161-096-from-w1"),
    "paper-loco-manip-161-032-amo": ("entities", "paper-loco-manip-161-135-amo"),
    "paper-loco-manip-161-036-falcon": ("entities", "paper-loco-manip-161-109-falcon"),
    "paper-loco-manip-161-059-demohlm": ("entities", "paper-loco-manip-161-136-demohlm"),
    "paper-loco-manip-161-064-hmc": ("entities", "paper-loco-manip-161-039-hmc"),
    "paper-loco-manip-161-066-hiwet": ("entities", "paper-loco-manip-161-041-hiwet"),
    "paper-loco-manip-161-130-humanoidexo": ("entities", "paper-loco-manip-161-067-humanoidexo"),
    "paper-loco-manip-161-143-act": ("entities", "paper-loco-manip-161-131-open-television"),
    # URL-matched cross-survey (no 同题深读 yet)
    "paper-loco-manip-161-029-n029": (
        "entities",
        "paper-hrl-stack-07-learning_human_to_humanoid_real_time",
    ),
    "paper-loco-manip-161-031-n031": (
        "entities",
        "paper-hrl-stack-14-robust_and_generalized_humanoid_moti",
    ),
    "paper-loco-manip-161-030-n030": ("methods", "any2track"),
    "paper-loco-manip-161-047-thor": ("entities", "paper-hrl-stack-42-thor"),
    "paper-loco-manip-161-049-viral": ("entities", "paper-hrl-stack-28-viral"),
    "paper-loco-manip-161-053-n053": (
        "entities",
        "paper-hrl-stack-29-opening_the_sim_to_real_door_for_hum",
    ),
    "paper-loco-manip-161-054-n054": (
        "entities",
        "paper-amp-survey-07-adversarial_locomotion_and_motion_im",
    ),
    "paper-loco-manip-161-068-humanoidmimicgen": ("entities", "paper-humanoidmimicgen"),
    "paper-loco-manip-161-081-viral": ("entities", "paper-hrl-stack-28-viral"),
    "paper-loco-manip-161-089-n089": (
        "entities",
        "paper-hrl-stack-29-opening_the_sim_to_real_door_for_hum",
    ),
    "paper-loco-manip-161-090-n090": (
        "entities",
        "paper-hrl-stack-33-ego_vision_world_model_for_humanoid",
    ),
    "paper-loco-manip-161-113-humanoid": ("entities", "paper-amp-survey-13-humanoid_goalkeeper"),
    "paper-loco-manip-161-127-n127": (
        "entities",
        "paper-hrl-stack-33-ego_vision_world_model_for_humanoid",
    ),
    "paper-loco-manip-161-134-wt-umi": ("entities", "paper-loco-manip-07-wt-umi"),
    # motion-cerebellum survey stub
    "paper-motion-cerebellum-humanoidmimicgen": ("entities", "paper-humanoidmimicgen"),
}

SKIP_GLOBS = {
    ".git",
    "node_modules",
    ".cursor-artifacts",
    "sources/raw",
}


def stub_num(stub: str) -> int | None:
    m = re.match(r"paper-loco-manip-161-(\d{3})-", stub)
    return int(m.group(1)) if m else None


def replace_links(text: str, stub: str, subdir: str, canon: str) -> str:
    """Replace markdown paths; avoid blind substring replace on the stub slug."""
    patterns = [
        (rf"\.\./entities/{re.escape(stub)}", f"../{subdir}/{canon}"),
        (rf"\./{re.escape(stub)}", f"./{canon}"),
        (rf"\.\./\.\./wiki/entities/{re.escape(stub)}", f"../../wiki/{subdir}/{canon}"),
        (rf"wiki/entities/{re.escape(stub)}", f"wiki/{subdir}/{canon}"),
        (rf"\({re.escape(stub)}\.md\)", f"({canon}.md)"),
        (rf"/{re.escape(stub)}\.md", f"/{canon}.md"),
    ]
    for pat, repl in patterns:
        text = re.sub(pat, repl, text)
    return text


def walk_text_files() -> list[Path]:
    out: list[Path] = []
    for base in (ROOT,):
        for p in base.rglob("*"):
            if not p.is_file():
                continue
            if any(part in SKIP_GLOBS for part in p.parts):
                continue
            if p.suffix in {".md", ".py", ".yml", ".json", ".html", ".js"}:
                out.append(p)
    return out


def update_bootstrap(nums: dict[int, str]) -> None:
    text = BOOTSTRAP.read_text(encoding="utf-8")
    block = "CANONICAL_ENTITY_BY_NUM: dict[int, str] = {\n"
    lines = [block]
    for num in sorted(nums):
        lines.append(f'    {num}: "{nums[num]}",\n')
    lines.append("}\n")
    new_block = "".join(lines)
    m = re.search(
        r"CANONICAL_ENTITY_BY_NUM: dict\[int, str\] = \{.*?\n\}\n",
        text,
        flags=re.S,
    )
    if not m:
        raise SystemExit("CANONICAL_ENTITY_BY_NUM block not found in bootstrap script")
    text = text[: m.start()] + new_block + text[m.end() :]
    BOOTSTRAP.write_text(text, encoding="utf-8")


def main() -> None:
    deleted: list[str] = []
    nums: dict[int, str] = {}

    # existing canonical nums from bootstrap CANONICAL_ENTITY_BY_NUM block only
    boot = BOOTSTRAP.read_text(encoding="utf-8")
    block_m = re.search(
        r"CANONICAL_ENTITY_BY_NUM: dict\[int, str\] = \{(.*?)\n\}",
        boot,
        flags=re.S,
    )
    if block_m:
        for m in re.finditer(r"^\s*(\d+):\s*\"([^\"]+)\"", block_m.group(1), re.M):
            nums[int(m.group(1))] = m.group(2)

    for stub, (subdir, canon) in sorted(MERGE_MAP.items()):
        path = ENTITIES / f"{stub}.md"
        if path.exists():
            path.unlink()
            deleted.append(stub)
        n = stub_num(stub)
        if n is not None and not canon.startswith("paper-loco-manip-161-"):
            nums[n] = canon

    changed_files = 0
    for fp in walk_text_files():
        if fp.resolve() == Path(__file__).resolve():
            continue
        if fp.resolve() == BOOTSTRAP.resolve():
            continue
        try:
            text = fp.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        orig = text
        for stub, (subdir, canon) in MERGE_MAP.items():
            if stub in text:
                text = replace_links(text, stub, subdir, canon)
        if text != orig:
            fp.write_text(text, encoding="utf-8")
            changed_files += 1

    if CATALOG.exists():
        cat = CATALOG.read_text(encoding="utf-8")
        cat = cat.replace(
            "每篇** 对应独立实体 `wiki/entities/paper-loco-manip-161-{NNN}-*.md`。",
            "每篇** 对应 wiki 实体；与 42 篇栈 / BFM / AMP / loco-manip-8 等姊妹篇重叠时 **链接到既有 canonical 实体**（见 `scripts/dedupe_loco_manip_161_entities.py`）。",
        )
        CATALOG.write_text(cat, encoding="utf-8")

    update_bootstrap(nums)

    print(f"deleted stubs: {len(deleted)}")
    for s in deleted:
        print(f"  - {s}")
    print(f"updated files: {changed_files}")
    print(f"canonical slots in bootstrap: {len(nums)}")


if __name__ == "__main__":
    main()
