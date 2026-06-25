#!/usr/bin/env python3
"""Deepen survey stub wiki entities (策展摘要 → compiled entity pages with 核心机制)."""

from __future__ import annotations

import importlib.util
import re
import sys
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENTITIES = ROOT / "wiki" / "entities"
TODAY = date.today().isoformat()

RAW_HRL = ROOT / "sources/raw/wechat_humanoid_rl_42_survey_2026-05-26.md"
RAW_MC = ROOT / "sources/raw/wechat_motion_cerebellum_64_survey_2026-06-18.md"
RAW_LOCO = ROOT / "sources/raw/wechat_loco_manip_8_papers_2026-06-14.md"
RAW_EGO = ROOT / "sources/raw/wechat_ego_9_papers_2026-06-01.md"

WM_ROUTE_MECH = {
    "01": "先预测未来视觉/潜特征（视频、RGB-D、法线等），再由独立或轻量动作头解码控制指令；误差在级联各段传递，工程上易拆模块复用。",
    "02": "未来观测与动作在同一扩散/自回归骨干中联合建模，减少级联误差累积；适合端到端 VLA/操控闭环。",
    "03": "世界模型作为 RL/评估虚拟环境，在想象中 rollout 替代昂贵真机试错；强调物理一致性与下游策略增益。",
}

BFM_GROUP_MECH = {
    "forward-backward": [
        "在无监督或多任务 RL 中学习 **forward-backward（FB）** 或 successor 结构，把异构任务压进可调用的身体潜空间。",
        "上层通过 **目标姿态、奖励向量或 latent prompt** 在潜空间中检索/组合行为，而非为每个技能单独训练策略。",
        "与单技能 motion tracking 对照：BFM 关心 **覆盖面与可组合性**，不只单一参考跟踪成功率。",
    ],
    "goal-conditioned": [
        "以 **goal / reference / command** 为条件训练全身跟踪或交互策略，扩展人形可执行动作库。",
        "数据侧常融合 MoCap、视频、遥操作与 HOI；控制侧强调 **抗扰、恢复与跨参考泛化**。",
        "在 BFM taxonomy 中回答「身体能覆盖多少目标条件技能」。",
    ],
    "intrinsic-reward": [
        "无明确外部任务时，用 **intrinsic reward**（探索、多样性、后继态等）预训练身体策略。",
        "为后续 goal-conditioned 或 imitation 提供 **可迁移的探索先验**，降低冷启动样本需求。",
    ],
    "adaptation": [
        "在预训练 BFM 上通过 **task token、动力学适配或少量示范** 快速迁移到新任务/新机体。",
        "核心问题是 **保留基座能力的同时** 以低成本吸收新约束，而非从零重训。",
    ],
    "hierarchical": [
        "语言、VLA、扩散或规划器作为上层，**调用** 已封装的底层全身能力（tracking / WBC / latent skill）。",
        "接口设计（命令空间、时序、安全层）决定上层智能能否稳定使用身体。",
    ],
}

BFM_MISCONCEPTIONS = {
    "forward-backward": "BFM-Zero 类工作不是「更大动作数据集」本身，而是 **潜空间可被 prompt 检索** 的身体接口。",
    "goal-conditioned": "Goal-conditioned 跟踪不等于 unlimited skills：仍受数据分布、接触建模与实机 Sim2Real 约束。",
    "intrinsic-reward": "Intrinsic 预训练不替代任务奖励；它提供 **探索覆盖**，下游仍需任务或示范对齐。",
    "adaptation": "Fast adaptation 论文通常假设 **已有强预训练基座**；弱基座上 adapter 收益有限。",
    "hierarchical": "语言/VLA 调用身体时，瓶颈往往在 **底层跟踪鲁棒性**，而非上层 token 设计 alone。",
}

HRL_LAYER_MISCON = {
    "data": "重定向/遥操作不是「训练前脚本」——参考质量上限往往 **早于** RL 策略决定。",
    "tracking-control": "Motion tracking 论文的泛化常指 **参考分布内**；换数据源或接触条件仍可能崩塌。",
    "perception": "感知 locomotion 的难点在 **闭环时延与几何误差**，不是单纯「加相机输入」。",
    "task-world": "VLA/世界模型条目解决 **接口与预测**，不自动替代已封装的底层 WBC 能力。",
    "contact-safety": "柔顺/恢复策略要在 **接触丰富** 与 **长期稳定** 间折中，不能只看单帧姿态。",
}

DEEP_READ_LINKS: dict[str, str] = {
    "paper-sonic.md": "../methods/sonic-motion-tracking.md",
    "paper-beyondmimic.md": "../methods/beyondmimic.md",
    "paper-opentrack.md": "../methods/any2track.md",
    "paper-ams.md": "../methods/ams.md",
    "paper-bfm-zero.md": "../entities/paper-behavior-foundation-model-humanoid.md",
    "paper-twist2.md": "../entities/paper-twist2.md",
    "paper-hrl-stack-11-deepmimic.md": "../methods/deepmimic.md",
    "paper-hrl-stack-01-retargeting_matters.md": "../methods/motion-retargeting-gmr.md",
    "paper-hrl-stack-02-make_tracking_easy.md": "../methods/neural-motion-retargeting-nmr.md",
    "paper-hrl-stack-28-viral.md": "../entities/paper-viral-humanoid-visual-sim2real.md",
    "paper-hrl-stack-29-opening_the_sim_to_real_door_for_hum.md": "../entities/paper-doorman-opening-sim2real-door.md",
    "paper-hrl-stack-34-gr00t_n1.md": "../entities/gr00t-wholebodycontrol.md",
    "paper-hrl-stack-37-gentlehumanoid.md": "../methods/gentlehumanoid-motion-tracking.md",
    "paper-hrl-stack-40-heracles.md": "../entities/paper-heracles-humanoid-diffusion.md",
    "paper-hrl-stack-21-adaptive-humanoid-control.md": "../entities/paper-adaptive-humanoid-control.md",
    "paper-adaptive-humanoid-control.md": "../entities/paper-adaptive-humanoid-control.md",
    "paper-hrl-stack-23-deep_whole_body_parkour.md": "../entities/paper-deep-whole-body-parkour.md",
    "paper-deep-whole-body-parkour.md": "../entities/paper-deep-whole-body-parkour.md",
    "paper-hiking-in-the-wild.md": "../entities/paper-hiking-in-the-wild.md",
}


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_stack = _load_module("stack_survey", ROOT / "scripts/generate_humanoid_stack_survey.py")
_bfm = _load_module("bfm_sources", ROOT / "scripts/generate_bfm_awesome_sources.py")
_vln = _load_module("vln_ingest", ROOT / "scripts/gen_vln_10_papers_ingest.py")

HRL_LAYER = _stack.HRL_LAYER
_parse_hrl_basic = _stack._parse_hrl
BFM_PAPERS = _bfm.PAPERS
GROUP_LABEL = _bfm.GROUP_LABEL
VLN_PAPERS = _vln.PAPERS


@dataclass
class Enrichment:
    one_liner: str = ""
    why_bullets: list[str] = field(default_factory=list)
    mechanism_bullets: list[str] = field(default_factory=list)
    misconceptions: list[str] = field(default_factory=list)
    deep_read: str | None = None
    experiment_note: str = ""


def _strip_images(text: str) -> str:
    text = re.sub(r"!\[[^\]]*\]\([^)]+\)[^\n]*", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _split_frontmatter(text: str) -> tuple[str, str]:
    if not text.startswith("---\n"):
        return "", text
    end = text.find("\n---\n", 4)
    if end < 0:
        return "", text
    return text[4:end], text[end + 5 :]


def _fm_get(fm: str, key: str) -> str:
    m = re.search(rf"^{key}:\s*(.+)$", fm, re.MULTILINE)
    if not m:
        return ""
    return m.group(1).strip().strip('"')


def _fm_summary(fm: str) -> str:
    m = re.search(r'^summary:\s*"(.+)"', fm, re.MULTILINE | re.DOTALL)
    if m:
        return m.group(1).strip()
    return _fm_get(fm, "summary")


def _update_fm_date(fm: str) -> str:
    if re.search(r"^updated:", fm, re.MULTILINE):
        return re.sub(r"^updated:.*$", f"updated: {TODAY}", fm, count=1, flags=re.MULTILINE)
    return fm.rstrip() + f"\nupdated: {TODAY}"


def _parse_hrl_extended(raw: str) -> dict[int, dict]:
    basic = {p["num"]: p for p in _parse_hrl_basic(raw)}
    parts = re.split(r"(?=^### \d+)", raw, flags=re.MULTILINE)
    for sec in parts:
        hm = re.match(r"^### (\d+)", sec)
        if not hm:
            continue
        num = int(hm.group(1))
        if num not in basic:
            continue
        paras: list[str] = []
        for block in re.split(r"\n\n+", sec):
            block = block.strip()
            if not block or block.startswith("!["):
                continue
            if re.match(r"^### \d+", block):
                continue
            if block.startswith("🔗") or block.startswith("📄") or block.startswith("🏫"):
                continue
            if len(block) > 40 and not block.startswith("|"):
                cleaned = _strip_images(block)
                if cleaned and not cleaned.startswith("•"):
                    paras.append(cleaned)
        basic[num]["paragraphs"] = paras
    return basic


def _build_hrl_entity_map() -> dict[str, int]:
    catalog = (ROOT / "sources/papers/humanoid_rl_stack_42_catalog.md").read_text(encoding="utf-8")
    out: dict[str, int] = {}
    for line in catalog.splitlines():
        m = re.search(r"\|\s*(\d+)\s*\|[^|]+\|[^|]+\|\s*\[\.\./\.\./wiki/entities/([^\]]+)\]", line)
        if m:
            out[m.group(2)] = int(m.group(1))
    return out


def _parse_motion_cerebellum(raw: str) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for m in re.finditer(
        r"^(\d+)\.\s+([^｜\n]+)[｜|]([^\n]+)\n+(.*?)(?=^\d+\.\s+|\Z)",
        raw,
        re.MULTILINE | re.DOTALL,
    ):
        num, short, tagline, body = m.group(1), m.group(2).strip(), m.group(3).strip(), m.group(4)
        reason = ""
        rm = re.search(r"\*\*我把它放在这里的原因：\*\*(.+)", body)
        if rm:
            reason = rm.group(1).strip()
        inst = ""
        im = re.search(r"\*\*发表机构：\*\*(.+)", body)
        if im:
            inst = im.group(1).strip()
        link = ""
        lm = re.search(r"\*\*项目 / 论文页：\*\*(https?://\S+)", body)
        if lm:
            link = lm.group(1).strip()
        key = re.sub(r"[^a-z0-9]+", "-", short.lower()).strip("-")
        out[key] = {
            "num": num,
            "short": short,
            "tagline": tagline,
            "reason": reason or tagline,
            "inst": inst,
            "link": link,
        }
    return out


def _parse_loco_manip(raw: str) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for m in re.finditer(
        r"^(\d+)\.\s+\*\*([^*]+)\*\*（([^)]+)）：(.+)$",
        raw,
        re.MULTILINE,
    ):
        num, short, meta, summary = m.group(1), m.group(2).strip(), m.group(3), m.group(4).strip()
        slug = re.sub(r"[^a-z0-9]+", "-", short.lower()).strip("-")
        out[slug] = {"num": num, "short": short, "meta": meta, "summary": summary}
    return out


def _parse_ego(raw: str) -> dict[int, dict]:
    out: dict[int, dict] = {}
    parts = re.split(r"(?=论文\s+\d+)", raw)
    for sec in parts:
        hm = re.match(r"论文\s+(\d+)([^\n]+)", sec)
        if not hm:
            continue
        num = int(hm.group(1))
        paras: list[str] = []
        title = ""
        tm = re.search(r"📄\s*论文标题：(.+)", sec)
        if tm:
            title = tm.group(1).strip()
        for block in re.split(r"\n\n+", sec):
            block = _strip_images(block.strip())
            if len(block) < 50 or block.startswith("📄") or block.startswith("🏛"):
                continue
            if block.startswith("|") or block.startswith("论文 "):
                continue
            paras.append(block)
        out[num] = {"title": title, "paragraphs": paras[:4]}
    return out


def _parse_wm_sources() -> dict[int, dict]:
    out: dict[int, dict] = {}
    for path in sorted((ROOT / "sources/papers").glob("shenlan_wm_survey_*.md")):
        m = re.search(r"shenlan_wm_survey_(\d+)_", path.name)
        if not m:
            continue
        num = int(m.group(1))
        text = path.read_text(encoding="utf-8")
        one = ""
        om = re.search(r"\*\*一句话说明：\*\*\s*(.+)", text)
        if om:
            one = om.group(1).strip()
        route = ""
        rm = re.search(r"\*\*路线分类：\*\*\s*(\d+)", text)
        if rm:
            route = rm.group(1)
        title = path.read_text(encoding="utf-8").splitlines()[0].lstrip("# ").strip()
        out[num] = {"title": title, "one_liner": one, "route": route}
    return out


def _extract_abbr_table(body: str) -> str | None:
    m = re.search(r"(## 英文缩写速查\n\n(?:\|[^\n]+\n)+)", body)
    return m.group(1).rstrip() if m else None


def _default_abbr(stack: str) -> str:
    rows = {
        "hrl": [
            ("RL", "Reinforcement Learning", "通过与环境交互最大化长期回报来学习策略的范式"),
            ("WBC", "Whole-Body Control", "协调全身关节满足多任务/约束的控制层"),
            ("Sim2Real", "Simulation to Reality", "从仿真训练迁移到真实机器人的技术总称"),
        ],
        "bfm": [
            ("BFM", "Behavior Foundation Model", "大规模行为数据预训练的可复用全身行为先验"),
            ("RL", "Reinforcement Learning", "通过与环境交互学习策略的范式"),
            ("WBC", "Whole-Body Control", "全身协调控制层"),
        ],
        "vln": [
            ("VLN", "Vision-and-Language Navigation", "依据自然语言指令在环境中导航的具身任务"),
            ("VLM", "Vision-Language Model", "视觉-语言多模态大模型"),
            ("SR", "Success Rate", "导航任务到达目标的成功率指标"),
        ],
        "wm": [
            ("WM", "World Model", "学习环境动态以供想象/规划的世界模型"),
            ("VLA", "Vision-Language-Action", "视觉-语言-动作端到端策略模型"),
            ("RGB-D", "RGB + Depth", "彩色图与深度图联合感知"),
        ],
        "ego": [
            ("Ego", "Egocentric", "佩戴式第一视角观察与数据采集"),
            ("VLA", "Vision-Language-Action", "视觉-语言-动作端到端策略"),
            ("MoCap", "Motion Capture", "动作捕捉与人体运动重建"),
        ],
        "loco": [
            ("Loco-Manip", "Loco-Manipulation", "行走与操作动力学耦合的全身任务"),
            ("WBC", "Whole-Body Control", "全身协调控制层"),
            ("Sim2Real", "Simulation to Reality", "仿真到真机迁移"),
        ],
        "mc": [
            ("WBC", "Whole-Body Control", "协调全身关节满足多任务/约束的控制层"),
            ("RL", "Reinforcement Learning", "通过与环境交互学习策略的范式"),
            ("Loco-Manip", "Loco-Manipulation", "移动操作耦合任务"),
        ],
    }
    lines = [
        "## 英文缩写速查",
        "",
        "| 缩写 | 英文全称 | 简要说明 |",
        "|------|----------|----------|",
    ]
    for abbr, full, desc in rows.get(stack, rows["hrl"]):
        lines.append(f"| {abbr} | {full} | {desc} |")
    return "\n".join(lines)


def _h1(body: str) -> str:
    m = re.search(r"^#\s+(.+)$", body, re.MULTILINE)
    return m.group(1).strip() if m else "Untitled"


def _intro_line(body: str) -> str:
    m = re.search(r"^#\s+.+\n\n(\*\*.+?\*\*[^\n]+)", body, re.MULTILINE | re.DOTALL)
    if not m:
        return ""
    line = m.group(1)
    line = line.replace("本页为知识库 **策展摘要**；方法细节以论文 PDF 与项目页为准。", "")
    line = re.sub(r"本页为知识库 \*\*策展摘要\*\*[^。]*。", "", line)
    line = re.sub(r"本页为 \*\*策展索引级\*\*[^。]*。", "", line)
    return line.strip()


def is_stub(text: str) -> bool:
    if "## 核心机制" in text:
        return False
    if "本页为知识库 **策展摘要**" in text:
        return True
    if "本页为 **策展索引级**" in text and "type: entity" in text:
        return True
    return False


def should_skip(path: Path, text: str) -> bool:
    if "paper-notebook-" in path.name and re.search(r"^status:\s*stub\b", text, re.MULTILINE):
        return True
    return False


class Deepener:
    def __init__(self) -> None:
        self.hrl_raw = (
            _parse_hrl_extended(RAW_HRL.read_text(encoding="utf-8")) if RAW_HRL.exists() else {}
        )
        self.hrl_map = _build_hrl_entity_map()
        self.mc_raw = (
            _parse_motion_cerebellum(RAW_MC.read_text(encoding="utf-8")) if RAW_MC.exists() else {}
        )
        self.loco_raw = (
            _parse_loco_manip(RAW_LOCO.read_text(encoding="utf-8")) if RAW_LOCO.exists() else {}
        )
        self.ego_raw = _parse_ego(RAW_EGO.read_text(encoding="utf-8")) if RAW_EGO.exists() else {}
        self.wm_raw = _parse_wm_sources()
        self.bfm_by_entity = {}
        for p in BFM_PAPERS:
            slug = p["slug"].split("_arxiv")[0].split("_icml")[0].split("_neurips")[0]
            slug = slug.replace("_", "-")
            if p["id"] == 13:
                continue
            self.bfm_by_entity[
                f"paper-bfm-{p['id']:02d}-{slug.replace('bfm-', '').replace('bfm_', '')}.md"
            ] = p
            ent = ROOT / "scripts/generate_bfm_awesome_wiki_entities.py"
            mod = _load_module("bfm_ent", ent)
            en = mod.paper_entity_name(p)
            if en:
                self.bfm_by_entity[en] = p
        self.vln_by_entity = {f"paper-vln-{p['num']}-{p['slug']}.md": p for p in VLN_PAPERS}

    def enrich(self, path: Path, fm: str, body: str) -> Enrichment | None:
        name = path.name
        e = Enrichment()
        summary = _fm_summary(fm)

        if name in DEEP_READ_LINKS:
            e.deep_read = DEEP_READ_LINKS[name]

        if name.startswith("paper-hrl-stack-") or name in self.hrl_map:
            num = self.hrl_map.get(name)
            if num is None:
                m = re.match(r"paper-hrl-stack-(\d+)-", name)
                num = int(m.group(1)) if m else None
            if num and num in self.hrl_raw:
                p = self.hrl_raw[num]
                e.one_liner = p.get("note") or summary
                e.why_bullets = [
                    f"在 [人形 RL 身体系统栈](../overview/humanoid-rl-motion-control-body-system-stack.md) 中属于 **{HRL_LAYER[p['layer']]}**（#{num:02d}/42）。",
                    p.get("note", ""),
                ]
                for para in p.get("paragraphs", [])[:3]:
                    if para not in e.why_bullets:
                        e.why_bullets.append(para)
                e.mechanism_bullets = p.get("paragraphs", [])[:4] or [p.get("note", "")]
                e.misconceptions = [HRL_LAYER_MISCON.get(p["layer"], "")]
                if num in _stack.HRL_WIKI_HINTS:
                    rel = _stack._wiki_to_entity_rel(_stack.HRL_WIKI_HINTS[num][0])
                    e.deep_read = e.deep_read or rel
                return e

        standalone_hrl = {
            "paper-twist.md": 9,
            "paper-twist2.md": 10,
            "paper-opentrack.md": 13,
            "paper-beyondmimic.md": 15,
            "paper-sonic.md": 17,
            "paper-ams.md": 18,
            "paper-bfm-zero.md": 19,
            "paper-adaptive-humanoid-control.md": 21,
            "paper-deep-whole-body-parkour.md": 23,
            "paper-hiking-in-the-wild.md": 24,
        }
        if name in standalone_hrl:
            num = standalone_hrl[name]
            if num in self.hrl_raw:
                p = self.hrl_raw[num]
                e.one_liner = p.get("note") or summary
                e.why_bullets = [
                    p.get("note", ""),
                    f"42 篇栈 **#{num:02d}** · {HRL_LAYER[p['layer']]}。",
                ]
                e.mechanism_bullets = p.get("paragraphs", [])[:4] or [p.get("note", "")]
                e.misconceptions = [HRL_LAYER_MISCON.get(p["layer"], "")]
                return e

        if name in self.bfm_by_entity:
            p = self.bfm_by_entity[name]
            grp = p["group"]
            e.one_liner = p["note"] or summary
            e.why_bullets = [
                p["note"],
                f"在 [BFM 41 篇技术地图](../overview/bfm-41-papers-technology-map.md) 中属于 **{GROUP_LABEL[grp]}**（#{p['id']:02d}/41）。",
            ]
            e.mechanism_bullets = list(BFM_GROUP_MECH.get(grp, [p["note"]]))
            e.misconceptions = [BFM_MISCONCEPTIONS.get(grp, "")]
            for w in p.get("wiki", []):
                if w.startswith("wiki/methods/") or w.startswith("wiki/entities/paper-"):
                    rel = "../" + w[len("wiki/") :]
                    e.deep_read = e.deep_read or rel
            return e

        if name in self.vln_by_entity:
            p = self.vln_by_entity[name]
            e.one_liner = p["summary"]
            e.why_bullets = [p["why"], p["summary"]]
            e.mechanism_bullets = [
                f"**任务形式：** {p['summary']}",
                f"**机构/出处：** {p['inst']} · {p['venue']}",
                f"**在 VLN 地图中的位置：** {p['cat_name']}（#{p['num']}/10）。",
            ]
            e.misconceptions = [
                "VLN benchmark 提升不等于真机部署；连续环境 (VLN-CE) 与离散图设定不可直接混比。"
            ]
            return e

        if name.startswith("paper-shenlan-wm-"):
            m = re.match(r"paper-shenlan-wm-(\d+)-", name)
            if m:
                num = int(m.group(1))
                w = self.wm_raw.get(num, {})
                e.one_liner = w.get("one_liner") or summary
                route = w.get("route", "01")
                e.why_bullets = [
                    e.one_liner,
                    f"属于 [世界模型 15 项目地图](../overview/world-models-15-open-source-technology-map.md) **路线 {route}**。",
                ]
                e.mechanism_bullets = [WM_ROUTE_MECH.get(route, ""), e.one_liner]
                e.misconceptions = [
                    "开源 WM 项目的引用量与 **控制一致性/下游任务增益** 无简单线性关系；复现前需核对 License 与权重。",
                ]
                return e

        if name.startswith("paper-motion-cerebellum-"):
            slug = name.replace("paper-motion-cerebellum-", "").replace(".md", "")
            mc = self.mc_raw.get(slug)
            if not mc:
                # fuzzy: active-spatial-brain...
                for k, v in self.mc_raw.items():
                    if k in slug or slug in k:
                        mc = v
                        break
            if mc:
                e.one_liner = mc["reason"]
                e.why_bullets = [
                    mc["reason"],
                    f"运动小脑 64 篇 **#{mc['num']}/64** · {mc['tagline']}。",
                ]
                e.mechanism_bullets = [
                    mc["reason"],
                    f"机构：{mc['inst']}" if mc.get("inst") else mc["tagline"],
                ]
                e.misconceptions = [
                    "运动小脑条目解决 **身体层** 问题，不替代 VLA/世界模型的任务规划。"
                ]
                return e
            if summary:
                e.one_liner = summary
                e.why_bullets = [summary]
                e.mechanism_bullets = [summary]
                return e

        if name.startswith("paper-loco-manip-"):
            m = re.match(r"paper-loco-manip-\d+-(.+)\.md", name)
            slug = m.group(1) if m else ""
            loc = self.loco_raw.get(slug)
            if loc:
                e.one_liner = loc["summary"]
                e.why_bullets = [
                    loc["summary"],
                    f"Loco-Manip 8 篇 **#{loc['num']}/8** · {loc['meta']}。",
                ]
                e.mechanism_bullets = [loc["summary"]]
                e.misconceptions = [
                    "Loco-manip 数据/接口论文不自动解决 **底层 WBC 鲁棒性**；须与跟踪/接触控制对照。"
                ]
                return e

        if name.startswith("paper-ego-"):
            m = re.match(r"paper-ego-(\d+)-", name)
            if m:
                num = int(m.group(1))
                eg = self.ego_raw.get(num, {})
                e.one_liner = summary or (
                    eg.get("paragraphs", [""])[0][:200] if eg.get("paragraphs") else ""
                )
                e.why_bullets = eg.get("paragraphs", [])[:2] or [summary]
                e.mechanism_bullets = eg.get("paragraphs", [])[:3] or [summary]
                e.misconceptions = [
                    "Ego 视频不会天然等于机器人策略数据；须经过重建、对齐、重定向与物理过滤。"
                ]
                return e

        if summary:
            e.one_liner = summary
            e.why_bullets = [summary]
            e.mechanism_bullets = [summary]
            e.misconceptions = ["策展编译不能替代原文消融与实机协议；量化指标以 PDF 为准。"]
            return e
        return None


def _extract_section(old_body: str, heading: str) -> str | None:
    pat = rf"(## {re.escape(heading)}[^\n]*\n(?:.*?\n)*?)(?=\n## |\Z)"
    m = re.search(pat, old_body, re.DOTALL)
    return m.group(1).rstrip() if m else None


def _abbr_for_path(path: Path, old_body: str) -> str:
    stack = "hrl"
    if path.name.startswith("paper-bfm-"):
        stack = "bfm"
    elif path.name.startswith("paper-vln-"):
        stack = "vln"
    elif path.name.startswith("paper-shenlan-wm-"):
        stack = "wm"
    elif path.name.startswith("paper-ego-"):
        stack = "ego"
    elif path.name.startswith("paper-loco-manip-"):
        stack = "loco"
    elif path.name.startswith("paper-motion-cerebellum-"):
        stack = "mc"
    return _extract_abbr_table(old_body) or _default_abbr(stack)


def _render_body(path: Path, fm: str, old_body: str, e: Enrichment) -> str:
    h1 = _h1(old_body)
    intro = _intro_line(old_body)
    if not intro:
        intro = f"**{h1}**"
    intro = intro.rstrip("。") + "。"

    abbr = _abbr_for_path(path, old_body)
    core_info = _extract_section(old_body, "核心信息（索引级）") or _extract_section(
        old_body, "核心信息"
    )
    relations = _extract_section(old_body, "与其他页面的关系")
    refs = _extract_section(old_body, "参考来源")
    more = _extract_section(old_body, "推荐继续阅读")

    parts = [f"# {h1}", "", intro, ""]
    if e.deep_read:
        label = e.deep_read.split("/")[-1].replace(".md", "")
        parts.append(
            f"> **深读页：** [{label}]({e.deep_read}) — 方法机制与实验细节见链接页；本页保留 survey 坐标与交叉引用。"
        )
        parts.append("")

    parts.extend(["## 一句话定义", "", e.one_liner, "", abbr, "", "## 为什么重要", ""])
    seen: set[str] = set()
    for b in e.why_bullets:
        b = b.strip()
        if b and b not in seen:
            seen.add(b)
            parts.append(f"- {b}")
    parts.append("")

    if core_info:
        parts.append(core_info)
        parts.append("")

    parts.extend(["## 核心机制（归纳）", ""])
    for i, b in enumerate(e.mechanism_bullets, 1):
        b = b.strip()
        if b:
            parts.append(f"### {i}）策展导读要点")
            parts.append("")
            parts.append(b)
            parts.append("")

    parts.extend(["## 常见误区", ""])
    for i, m in enumerate(e.misconceptions, 1):
        m = m.strip()
        if m:
            parts.append(f"{i}. {m}")
    parts.append("")

    parts.extend(
        [
            "## 实验与评测",
            "",
            "- 本页在公众号/survey **策展编译**基础上补充机制归纳；**量化 benchmark、消融与实机指标以原文 PDF / 项目页为准**（链接见 [参考来源](#参考来源)）。",
            "- 与同栈姊妹篇对照时，请回到对应 **技术地图 / 42 篇栈 / BFM 地图 / VLN 地图** 总览中的实验段落。",
            "",
        ]
    )

    if relations:
        parts.append(relations.rstrip())
    else:
        parts.extend(
            [
                "## 与其他页面的关系",
                "",
                "- 见 frontmatter `related` 与 [参考来源](#参考来源)。",
            ]
        )
    parts.append("")

    if refs:
        parts.append(refs.rstrip())
    else:
        parts.extend(["## 参考来源", "", "- （见 frontmatter `sources`）"])
    if more:
        parts.append("")
        parts.append(more.rstrip())

    return "\n".join(parts) + "\n"


def deepen_file(path: Path, dry_run: bool = False) -> bool:
    text = path.read_text(encoding="utf-8")
    if not is_stub(text) or should_skip(path, text):
        return False
    fm, body = _split_frontmatter(text)
    if not fm:
        return False
    d = Deepener()
    e = d.enrich(path, fm, body)
    if not e or not e.one_liner:
        print(f"skip (no enrichment): {path.relative_to(ROOT)}", file=sys.stderr)
        return False
    new_fm = _update_fm_date(fm)
    new_body = _render_body(path, fm, body, e)
    out = f"---\n{new_fm}\n---\n\n{new_body}"
    if dry_run:
        print(f"would deepen: {path.relative_to(ROOT)}")
        return True
    path.write_text(out, encoding="utf-8")
    print(f"deepened: {path.relative_to(ROOT)}")
    return True


def main() -> None:
    dry = "--dry-run" in sys.argv
    paths = sorted(ENTITIES.glob("*.md"))
    n = 0
    for p in paths:
        if deepen_file(p, dry_run=dry):
            n += 1
    print(f"{'would deepen' if dry else 'deepened'} {n} entity pages")


if __name__ == "__main__":
    main()
