#!/usr/bin/env python3
"""从 wiki 正文 |机构| 表、sources 机构行、GitHub org 与覆盖表同步 frontmatter tags。

与 bump_institution_tags.py 互补：本脚本侧重结构化来源（表格 / sources），
bump 脚本侧重摘要区关键词推断。二者可串联执行。

用法：
  python3 scripts/sync_institution_tags.py --dry-run
  python3 scripts/sync_institution_tags.py
  python3 scripts/sync_institution_tags.py --entities-only
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
WIKI_DIR = REPO_ROOT / "wiki"

sys.path.insert(0, str(REPO_ROOT / "scripts"))
from bump_institution_tags import (  # noqa: E402
    INSTITUTIONS_PATH,
    PAGE_INSTITUTION_OVERRIDES,
    _add_tags_to_frontmatter,
    _build_alias_map,
    _derive_institutions,
    _load_registry,
    _parse_frontmatter_list,
    _preferred_alias,
    is_tool_entity,
)

GITHUB_ORG_MAP: dict[str, str] = {
    "facebookresearch": "meta",
    "NVlabs": "nvidia",
    "nv-tlabs": "nvidia",
    "nvidia": "nvidia",
    "openai": "openai",
    "rll-research": "berkeley",
    "LeCAR-Lab": "cmu",
    "OpenRobotLab": "shanghai-ai-lab",
    "leggedrobotics": "eth",
    "mit-acl": "mit",
    "mit-biomimetics": "mit",
    "google-research": "google",
    "humanoid-clone": "stanford",
    "ZhengyiLuo": "berkeley",
    "Sirui-Xu": "sjtu",
    "GuyTevet": "meta",
    "liangpan99": "nus",
    "physical-intelligence": "physical-intelligence",
    "google-deepmind": "google-deepmind",
    "frankaemika": "franka",
    "bulletphysics": "pybullet",
    "robotis-git": "robotis",
    "ROBOTIS-GIT": "robotis",
    "open-dynamic-robot-initiative": "odri",
    "ODRI": "odri",
    "duckietown": "disney",
    "AntonOsika": "open-duck",
    "pybullet": "pybullet",
    "moveit": "linux-foundation",
    "picknikrobotics": "linux-foundation",
    "ros-planning": "linux-foundation",
    "ros": "linux-foundation",
    "turtlebot": "robotis",
    "jd-opensource": "jd",
}

ORG_PHRASE_MAP: list[tuple[str, str]] = [
    ("北京通用人工智能研究院", "bigai"),
    ("中关村人工智能研究院", "zgca"),
    ("上海人工智能实验室", "shanghai-ai-lab"),
    ("清华大学交叉信息研究院", "tsinghua"),
    ("清华大学计算机系", "tsinghua"),
    ("中国电信人工智能研究院", "teleai"),
    ("中国电信 teleai", "teleai"),
    ("中国电信", "teleai"),
    ("上海期智研究院", "shanghai-pil"),
    ("上海科技大学", "shanghaitech"),
    ("哈尔滨工业大学", "hit"),
    ("哈尔滨工程大学", "heu"),
    ("西北工业大学深圳研究院", "npu"),
    ("西北工业大学", "npu"),
    ("中国科学技术大学", "ustc"),
    ("中国科学院大学", "ucas"),
    ("宁波东方理工大学", "ningbo-eastern"),
    ("北京航空航天大学", "buaa"),
    ("北京工业大学", "bjut"),
    ("北京理工大学", "bit"),
    ("北京智源人工智能研究院", "baai"),
    ("人形机器人（上海）有限公司", "openloong"),
    ("北京人形机器人创新中心", "x-humanoid"),
    ("香港科技大学（广州）", "hkust-gz"),
    ("香港科技大学", "hkust"),
    ("香港中文大学", "cuhk"),
    ("香港理工大学", "polyu"),
    ("香港大学", "hku"),
    ("上海交通大学", "sjtu"),
    ("上海交大", "sjtu"),
    ("浙江大学", "zju"),
    ("南京大学", "nju"),
    ("复旦大学", "fudan"),
    ("中山大学", "sysu"),
    ("华南理工大学", "scut"),
    ("华中科技大学", "hust"),
    ("山东大学", "sdu"),
    ("中国农业大学", "cau"),
    ("天津大学", "tju"),
    ("北京大学", "pku"),
    ("清华大学", "tsinghua"),
    ("中科大", "ustc"),
    ("字节跳动 seed", "bytedance"),
    ("字节跳动", "bytedance"),
    ("智元机器人", "agibot"),
    ("逐际动力", "limx"),
    ("乐聚机器人", "leju"),
    ("小米机器人实验室", "xiaomi-robotics"),
    ("lumos robotics", "lumos"),
    ("opendrivelab", "opendrivelab"),
    ("法国国家信息与自动化研究所", "inria"),
    ("佐治亚理工学院", "georgia-tech"),
    ("georgia institute of technology", "georgia-tech"),
    ("俄勒冈州立大学", "oregon-state"),
    ("英属哥伦比亚大学", "ubc"),
    ("密歇根大学安娜堡分校", "umich"),
    ("密歇根大学", "umich"),
    ("华盛顿大学", "uw"),
    ("德州大学奥斯汀分校", "ut-austin"),
    ("ut austin", "ut-austin"),
    ("约翰内斯开普勒大学", "jku"),
    ("澳大利亚国立大学", "anu"),
    ("阿德莱德大学", "adelaide"),
    ("新加坡国立大学", "nus"),
    ("首尔大学", "snu"),
    ("加州理工学院", "caltech"),
    ("宾夕法尼亚大学", "upenn"),
    ("佛罗里达大学", "uf"),
    ("塔夫茨大学", "tufts"),
    ("杜克大学", "duke"),
    ("纽约大学", "nyu"),
    ("new york university", "nyu"),
    ("nyu shanghai", "nyu"),
    ("达姆施塔特工业大学", "tu-darmstadt"),
    ("technical university of darmstadt", "tu-darmstadt"),
    ("德国人工智能研究中心", "dfki"),
    ("dfki", "dfki"),
    ("马克斯·普朗克", "max-planck"),
    ("max planck", "max-planck"),
    ("人类与机器认知研究所", "caltech"),
    ("facebook ai research", "meta"),
    ("amazon far", "amazon"),
    ("simon fraser university", "sfu"),
    ("西蒙菲莎大学", "sfu"),
    ("西蒙弗雷泽大学", "sfu"),
    ("sea ai lab", "sea-ai-lab"),
    ("garena", "sea-ai-lab"),
    ("beingbeyond", "beingbeyond"),
    ("deepcybo", "deepcybo"),
    ("zozo, inc", "zozo"),
    ("zozo", "zozo"),
    ("hessian.ai", "tu-darmstadt"),
    ("robotics institute germany", "dfki"),
    ("英伟达研究院", "nvidia"),
    ("英伟达", "nvidia"),
    ("斯坦福大学", "stanford"),
    ("伯克利", "berkeley"),
    ("华为", "huawei"),
    ("bigai", "bigai"),
    ("kaist", "kaist"),
    ("usc", "usc"),
    ("sony", "sony"),
    ("snap", "snap"),
    ("meta", "meta"),
    ("nvidia", "nvidia"),
    ("cmu", "cmu"),
    ("stanford", "stanford"),
    ("berkeley", "berkeley"),
    ("uc berkeley", "berkeley"),
    ("mit", "mit"),
    ("eth zürich", "eth"),
    ("eth zurich", "eth"),
    ("苏黎世联邦理工", "eth"),
    ("princeton", "princeton"),
    ("google deepmind", "google-deepmind"),
    ("deepmind", "google-deepmind"),
    ("microsoft", "microsoft"),
    ("openai", "openai"),
    ("amazon", "amazon"),
    ("inria", "inria"),
    ("jdcloud", "jd"),
    ("京东", "jd"),
]
ORG_PHRASE_MAP.sort(key=lambda item: -len(item[0]))

SYNC_PAGE_OVERRIDES: dict[str, list[str]] = {
    "wiki/entities/1x-technologies.md": ["1x-technologies"],
    "wiki/entities/allegro-hand.md": ["wonik-robotics"],
    "wiki/entities/amass.md": ["max-planck"],
    "wiki/entities/dreamwaq-plus.md": ["kaist", "mit"],
    "wiki/entities/elephantrobotics-myagv.md": ["elephant-robotics"],
    "wiki/entities/elephantrobotics-mycobot-320.md": ["elephant-robotics"],
    "wiki/entities/figure-ai.md": ["figure-ai"],
    "wiki/entities/franka-research-3.md": ["franka"],
    "wiki/entities/generalist-ai-robotics.md": ["generalist-ai"],
    "wiki/entities/hands-on-rl-book.md": ["sjtu"],
    "wiki/entities/kinova-gen3.md": ["kinova"],
    "wiki/entities/linear-algebra-curriculum.md": ["georgia-tech"],
    "wiki/entities/mixamo.md": ["autodesk"],
    "wiki/entities/modern-robotics-book.md": ["northwestern"],
    "wiki/entities/moveit2.md": ["linux-foundation"],
    "wiki/entities/numerical-optimization-curriculum.md": ["stanford"],
    "wiki/entities/odri-solo-and-bolt.md": ["odri"],
    "wiki/entities/open-duck-mini-runtime.md": ["disney", "open-duck"],
    "wiki/entities/open-duck-mini.md": ["disney", "open-duck"],
    "wiki/entities/open-duck-playground.md": ["disney", "open-duck"],
    "wiki/entities/open-duck-reference-motion-generator.md": ["disney", "open-duck"],
    "wiki/entities/parol6-source-robotics.md": ["source-robotics"],
    "wiki/entities/pybullet.md": ["pybullet"],
    "wiki/entities/project-instinct.md": ["shanghai-pil", "tsinghua"],
    "wiki/entities/quadruped-control-curriculum.md": ["motrix"],
    "wiki/entities/qwen-robot-manip.md": ["alibaba"],
    "wiki/entities/qwen-robot-nav.md": ["alibaba"],
    "wiki/entities/qwen-robot-suite.md": ["alibaba"],
    "wiki/entities/qwen-robot-world.md": ["alibaba"],
    "wiki/entities/qwen-vla.md": ["alibaba"],
    "wiki/entities/robotis-open-manipulator-line.md": ["robotis"],
    "wiki/entities/robotis-op3.md": ["robotis"],
    "wiki/entities/robotis-thormang3.md": ["robotis"],
    "wiki/entities/robotwin.md": ["hku", "shanghai-ai-lab", "sjtu"],
    "wiki/entities/sceneverse-pp.md": ["bigai"],
    "wiki/entities/shadow-hand.md": ["shadow-robotics"],
    "wiki/entities/turtlebot3.md": ["robotis"],
    "wiki/entities/wuji-robotics.md": ["wuji-robotics"],
    "wiki/entities/xiaomi-robotics-0.md": ["xiaomi-robotics"],
    "wiki/entities/karpathy-autoresearch.md": ["karpathy"],
    "wiki/entities/sensenova-skills.md": ["sensenova"],
    "wiki/entities/dataset-bfm-amass.md": ["max-planck"],
    "wiki/entities/dataset-bfm-lafan.md": ["sfu"],
    "wiki/entities/dataset-bfm-kit-ml.md": ["kit"],
    "wiki/entities/dataset-bfm-babel.md": ["berkeley"],
    "wiki/entities/dataset-bfm-humanml3d.md": ["pku"],
    "wiki/entities/dataset-bfm-motion-x.md": ["pku"],
    "wiki/entities/dataset-bfm-motion-xpp.md": ["pku"],
    "wiki/entities/dataset-bfm-posescript.md": ["inria"],
    "wiki/entities/dataset-bfm-humanoid-x.md": ["pku"],
    "wiki/entities/lafan1-dataset.md": ["sfu"],
    "wiki/entities/robotic-world-model-eth-rsl.md": ["eth"],
    "wiki/entities/paper-amp-survey-12-haml.md": ["sdu"],
    "wiki/entities/paper-anymal-walk-minutes-parallel-drl.md": ["eth"],
    "wiki/entities/paper-bam-extended-friction-servo-actuators.md": ["google"],
    "wiki/entities/paper-bfm-03-fb-aw.md": ["meta"],
    "wiki/entities/paper-bfm-04-fast-imitation-bfm.md": ["meta"],
    "wiki/entities/paper-bfm-05-learning-one-representation.md": ["meta"],
    "wiki/entities/paper-bfm-06-successor-states.md": ["meta"],
    "wiki/entities/paper-bfm-16-modskill.md": ["hku", "upenn"],
    "wiki/entities/paper-bfm-20-moconvq.md": ["pku"],
    "wiki/entities/paper-bfm-21-case.md": ["hku"],
    "wiki/entities/paper-bfm-23-teamplay.md": ["google-deepmind"],
    "wiki/entities/paper-bfm-27-proto-rl.md": ["meta"],
    "wiki/entities/paper-bfm-28-re3.md": ["berkeley"],
    "wiki/entities/paper-bfm-30-diayn.md": ["berkeley"],
    "wiki/entities/paper-bfm-31-task-tokens.md": ["meta"],
    "wiki/entities/paper-bfm-32-unseen-dynamics.md": ["meta"],
    "wiki/entities/paper-bfm-33-fast-adaptation-bfm.md": ["meta"],
    "wiki/entities/paper-bfm-36-leverb.md": ["berkeley", "cmu"],
    "wiki/entities/paper-bfm-37-langwbc.md": ["berkeley"],
    "wiki/entities/paper-bfm-40-uniphys.md": ["cmu", "eth"],
    "wiki/entities/paper-cassie-biped-versatile-locomotion-rl.md": ["berkeley"],
    "wiki/entities/paper-cassie-feedback-control-drl.md": ["oregon-state", "ubc"],
    "wiki/entities/paper-cassie-iterative-locomotion-sim2real.md": ["oregon-state", "ubc"],
    "wiki/entities/paper-coins-compositional-human-scene-interaction.md": ["stanford"],
    "wiki/entities/paper-daji-anticipatory-joint-intent.md": ["hkust-gz", "limx", "sdu"],
    "wiki/entities/paper-deeprl-locomotion-action-space-sca2017.md": ["ubc"],
    "wiki/entities/paper-dimos-human-scene-motion-synthesis.md": ["eth", "google"],
    "wiki/entities/paper-ego-01-aoe.md": ["alibaba", "baai", "pku", "ucas", "zju"],
    "wiki/entities/paper-ego-02-egolive.md": ["jd"],
    "wiki/entities/paper-ego-03-egomimic.md": ["georgia-tech", "stanford"],
    "wiki/entities/paper-ego-04-emma.md": ["georgia-tech"],
    "wiki/entities/paper-ego-05-gaze2act.md": ["stanford"],
    "wiki/entities/paper-ego-08-egoexomem.md": ["eth"],
    "wiki/entities/paper-ego-09-e3c.md": ["meta"],
    "wiki/entities/paper-homeworld-whole-home-scene-generation.md": ["cuhk"],
    "wiki/entities/paper-humanoid-soccer-swarm-intelligence.md": ["sfu"],
    "wiki/entities/paper-learning-to-adapt-bio-inspired-quadruped-gait.md": ["purdue"],
    "wiki/entities/paper-loco-manip-07-wt-umi.md": ["georgia-tech"],
    "wiki/entities/paper-mamma-markerless-motion-capture.md": ["eth"],
    "wiki/entities/paper-mpc-rl-humanoid-locomotion-manipulation.md": ["eth"],
    "wiki/entities/paper-ppf-cubic-barrier-contact-solver.md": ["zozo"],
    "wiki/entities/paper-quadruped-torque-control-rl.md": ["berkeley"],
    "wiki/entities/paper-resnet-deep-residual-learning.md": ["microsoft"],
    "wiki/entities/paper-rma-rapid-motor-adaptation.md": ["berkeley", "cmu"],
    "wiki/entities/paper-shape-your-body-value-gradient-design.md": ["tu-darmstadt", "dfki"],
    "wiki/entities/paper-shenlan-wm-01-tesseract.md": ["hkust"],
    "wiki/entities/paper-shenlan-wm-02-vpp.md": [
        "berkeley",
        "shanghai-ai-lab",
        "shanghai-pil",
        "tsinghua",
    ],
    "wiki/entities/paper-shenlan-wm-03-lapa.md": ["ai2", "kaist", "microsoft", "nvidia", "uw"],
    "wiki/entities/paper-shenlan-wm-04-mimic-video.md": ["nvidia"],
    "wiki/entities/paper-shenlan-wm-05-villa-x.md": ["microsoft", "tsinghua"],
    "wiki/entities/paper-shenlan-wm-06-video-gen-robot-policies.md": ["toyota-research"],
    "wiki/entities/paper-shenlan-wm-07-worldvla.md": ["alibaba", "zju"],
    "wiki/entities/paper-shenlan-wm-08-uwm.md": ["toyota-research", "uw"],
    "wiki/entities/paper-shenlan-wm-09-gr1.md": ["bytedance"],
    "wiki/entities/paper-shenlan-wm-10-uva.md": ["stanford"],
    "wiki/entities/paper-shenlan-wm-11-cosmos-policy.md": ["nvidia", "stanford"],
    "wiki/entities/paper-shenlan-wm-12-f1-vla.md": ["hit", "shanghai-ai-lab"],
    "wiki/entities/paper-shenlan-wm-13-dreamerv3.md": ["google-deepmind"],
    "wiki/entities/paper-shenlan-wm-14-rlvr-world.md": ["tsinghua"],
    "wiki/entities/paper-shenlan-wm-15-worldgym.md": ["google-deepmind", "nyu", "stanford"],
    "wiki/entities/paper-variable-impedance-contact-rl.md": ["max-planck", "nyu"],
    "wiki/entities/paper-variable-stiffness-locomotion-rl.md": ["eth"],
    "wiki/entities/paper-vln-03-reverie.md": ["adelaide", "georgia-tech"],
    "wiki/entities/paper-walk-these-ways-quadruped-mob.md": ["mit"],
    "wiki/entities/paper-wem-world-ego-modeling.md": ["nvidia"],
    "wiki/entities/paper-worldvln-aerial-vln-wam.md": ["zju"],
    "wiki/entities/paper-yolo-unified-realtime-detection.md": ["ai2", "uw"],
    "wiki/entities/paper-htd-refine-monocular-hmr.md": ["eth"],
    "wiki/entities/paper-urdd-universal-robot-description-directory.md": ["linux-foundation"],
    "wiki/entities/motrix.md": ["motphys"],
    "wiki/entities/motioncode.md": ["motioncode"],
    "wiki/entities/current-robotics-curr0.md": ["current-robotics"],
    "wiki/entities/ruka-v2-hand.md": ["nyu"],
    "wiki/entities/smplolympics.md": ["cmu", "nvidia"],
}

SKIP_TABLE_VALUES = frozenset(
    {
        "内容",
        "—",
        "-",
        "待核对",
        "机构待核对",
        "首尔大学 朴在亨课题组（待最终核对）",
    }
)


def _merged_overrides() -> dict[str, list[str]]:
    merged = dict(PAGE_INSTITUTION_OVERRIDES)
    merged.update(SYNC_PAGE_OVERRIDES)
    return merged


def _map_phrases(text: str) -> list[str]:
    lowered = text.lower()
    found: list[str] = []
    seen: set[str] = set()
    for phrase, cid in ORG_PHRASE_MAP:
        if phrase.lower() in lowered and cid not in seen:
            seen.add(cid)
            found.append(cid)
    return found


def _resolve_source_path(src: str) -> Path | None:
    candidate = (REPO_ROOT / src.replace("../../", "")).resolve()
    if candidate.exists():
        return candidate
    alt = REPO_ROOT / "sources" / Path(src).name
    return alt if alt.exists() else None


def infer_institution_ids(rel_path: str, content: str, alias_map: dict[str, str]) -> list[str]:
    if _derive_institutions(content, alias_map):
        return []

    found: list[str] = []
    seen: set[str] = set()

    def add(cid: str) -> None:
        if cid and cid not in seen:
            seen.add(cid)
            found.append(cid)

    for cid in _merged_overrides().get(rel_path, []):
        add(cid)

    for match in re.finditer(r"\|\s*机构\s*\|\s*([^|\n]+)", content):
        value = match.group(1).strip()
        if not value or value in SKIP_TABLE_VALUES:
            continue
        if re.search(r"\d+\s*kg", value):
            continue
        for cid in _map_phrases(value):
            add(cid)

    for src in _parse_frontmatter_list(content, "sources"):
        sp = _resolve_source_path(src)
        if not sp:
            continue
        src_text = sp.read_text(encoding="utf-8", errors="ignore")
        for match in re.finditer(r"机构[：:]\s*([^\n]+)", src_text):
            val = match.group(1).strip()
            if val in SKIP_TABLE_VALUES:
                continue
            for cid in _map_phrases(val):
                add(cid)

    for org in re.findall(r"github\.com/([^/\s\"']+)", content):
        gh_cid = GITHUB_ORG_MAP.get(org)
        if gh_cid:
            add(gh_cid)

    return found


def _should_process(rel_path: str, tags: list[str], *, entities_only: bool) -> bool:
    if not rel_path.startswith("wiki/entities/"):
        return False
    if "paper-notebook" in Path(rel_path).stem:
        return False
    tagset = {t.lower() for t in tags}
    if "paper-notebook-stub" in tagset or "paper-notebook-planned" in tagset:
        return False
    if entities_only and not is_tool_entity(rel_path, tags):
        pass
    return True


def sync_page(
    path: Path,
    registry: dict[str, dict],
    alias_map: dict[str, str],
    *,
    dry_run: bool,
) -> list[str] | None:
    rel = path.relative_to(REPO_ROOT).as_posix()
    content = path.read_text(encoding="utf-8")
    if _derive_institutions(content, alias_map):
        return None

    cids = infer_institution_ids(rel, content, alias_map)
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
    parser = argparse.ArgumentParser(description="Sync institution tags from tables/sources/github")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--entities-only", action="store_true")
    args = parser.parse_args()

    registry = _load_registry(INSTITUTIONS_PATH)
    alias_map = _build_alias_map(registry)

    changed: list[tuple[str, list[str]]] = []
    for path in sorted(WIKI_DIR.rglob("*.md")):
        rel = path.relative_to(REPO_ROOT).as_posix()
        tags = _parse_frontmatter_list(path.read_text(encoding="utf-8"), "tags")
        if not _should_process(rel, tags, entities_only=args.entities_only):
            continue
        added = sync_page(path, registry, alias_map, dry_run=args.dry_run)
        if added:
            changed.append((rel, added))

    print(f"{'[dry-run] ' if args.dry_run else ''}updated {len(changed)} pages")
    for rel, tags in changed[:100]:
        print(f"  {rel}: +{tags}")
    if len(changed) > 100:
        print(f"  ... and {len(changed) - 100} more")
    return 0


if __name__ == "__main__":
    sys.exit(main())
