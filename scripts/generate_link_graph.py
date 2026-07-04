#!/usr/bin/env python3
"""
generate_link_graph.py — Wiki 内链图谱生成工具

扫描所有 wiki 页面的内链，生成 exports/link-graph.json，
供 docs/graph.html 的 D3.js 渲染使用。

同时写入 exports/graph-stats.json（含 latest_wiki_nodes：按 log.md 最新日历日
合并当日所有 `## [日期]` 块中出现的有效 wiki/... 路径（去重保序；ingest /
structural / query 等均可）；latest_wiki_node 为当日列表首项（兼容旧字段）。
若无日志命中则回退到 frontmatter / mtime 的 recency，列表仅一项。

另写入 exports/wiki-activity.json（首页热力图数据源）：不限时间窗口，按同一套
路径解析规则汇总 log.md 全量日志的每日 wiki 节点（同日去重、跨日可重复）。

输出格式：
  {
    "nodes": [
      {
        "id": "wiki/methods/mpc.md",
        "label": "MPC",
        "type": "method",
        "community": "community-0",
        "institutions": ["nvidia"]
      }
    ],
    "edges": [{"source": "wiki/methods/mpc.md", "target": "wiki/concepts/wbc.md"}],
    "communities": [{"id": "community-0", "label": "...", "size": 12}],
    "institutions": [{"id": "nvidia", "label": "英伟达（NVIDIA）", "size": 22}]
  }

用法：
  python3 scripts/generate_link_graph.py
  make graph
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from collections import Counter, defaultdict
from datetime import date, timedelta
from pathlib import Path
from typing import Any

from export_minimal import extract_summary
from utils.wiki_cache import wiki_stem_to_path

REPO_ROOT = Path(__file__).parent.parent
WIKI_DIR = REPO_ROOT / "wiki"
OUT_PATH = REPO_ROOT / "exports" / "link-graph.json"
STATS_PATH = REPO_ROOT / "exports" / "graph-stats.json"
ACTIVITY_PATH = REPO_ROOT / "exports" / "wiki-activity.json"
LOG_MD_PATH = REPO_ROOT / "log.md"
# log.md 正文中出现的 wiki 相对路径（允许省略 .md，匹配至非标点为止）
WIKI_PATH_IN_LOG = re.compile(r"wiki/(?:[\w./-]+/)+[\w./-]+(?:\.md)?", re.IGNORECASE)
# 反引号内的 wiki 通配路径，如 `wiki/entities/paper-bfm-*.md`
WIKI_GLOB_IN_LOG = re.compile(r"`(wiki/[^`]*\*(?:\.md)?)`", re.IGNORECASE)

# ── 研究机构注册表（schema/institutions.json）──────────────────────────────
# 单一事实源：机构 id → {label, aliases}。节点「所属机构」默认从 frontmatter
# tags 里精确匹配 alias 派生（一个节点可属于多个机构）；页面可用 frontmatter
# `institutions: [..]` 显式覆盖。新增机构只需改 JSON，不必动前端。
INSTITUTIONS_REGISTRY_PATH = REPO_ROOT / "schema" / "institutions.json"


def _load_institution_registry(path: Path) -> dict[str, dict[str, Any]]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    registry = data.get("registry", {})
    return registry if isinstance(registry, dict) else {}


def _build_institution_alias_map(registry: dict[str, dict[str, Any]]) -> dict[str, str]:
    """alias / canonical id（均小写）→ canonical id。"""
    alias_map: dict[str, str] = {}
    for canonical_id, meta in registry.items():
        alias_map[canonical_id.lower()] = canonical_id
        for alias in meta.get("aliases", []) or []:
            alias_map[str(alias).strip().lower()] = canonical_id
    return alias_map


INSTITUTION_REGISTRY: dict[str, dict[str, Any]] = _load_institution_registry(
    INSTITUTIONS_REGISTRY_PATH
)
INSTITUTION_ALIAS_MAP: dict[str, str] = _build_institution_alias_map(INSTITUTION_REGISTRY)

# 主社区检测（Louvain）合并后的目标社区数上限（与 MAX_COMMUNITIES 命名席位对齐）。
PRIMARY_COMMUNITY_CAP = 16
# 输出中显式命名的最多社区数：二级拆分后给细分社区更多席位，避免大量节点落入"其他社区"。
MAX_COMMUNITIES = 16
OTHER_COMMUNITY_ID = "community-other"
OTHER_COMMUNITY_LABEL = "其他（Other） 社区"
# 与同社区邻居的边占比低于此值的非枢纽节点归入「其他社区」（避免强行贴标签）。
COMMUNITY_MEMBERSHIP_THRESHOLD = 0.5
# 社区展示名格式：「中文（English） 社区」。规范见 schema/naming.md § 图谱社区命名。
# 社区基名默认取枢纽页 H1，但 H1 风格不一；此处按 hub 路径给出统一 override，脚本再追加 ` 社区`。
# 未命中 override 时回退 H1，并在 generate 阶段对不符合 COMMUNITY_HUB_NAME_RE 的基名打印 WARNING。
COMMUNITY_HUB_NAME_RE = re.compile(
    r"^[\u4e00-\u9fff]"  # 以中文开头
    r"[\u4e00-\u9fff\w\s·/·、，,\-：:]*"  # 中文主名（允许常见标点）
    r"（[^）]+）$"  # 全角括号内的英文/缩写副名
)
# 研究机构展示名与社区基名共用「中文（English）」格式（不含 ` 社区` 后缀）。规范见 schema/naming.md。
INSTITUTION_LABEL_RE = COMMUNITY_HUB_NAME_RE
COMMUNITY_NAME_OVERRIDES: dict[str, str] = {
    "wiki/overview/humanoid-rl-motion-control-body-system-stack.md": "人形强化学习运动控制（Humanoid Reinforcement Learning Motion Control, RL）",
    "wiki/concepts/whole-body-control.md": "全身控制（Whole-Body Control, WBC）",
    "wiki/methods/imitation-learning.md": "模仿学习（Imitation Learning, IL）",
    "wiki/tasks/locomotion.md": "运动控制（Locomotion）",
    "wiki/concepts/sim2real.md": "仿真到现实（Simulation to Reality, Sim2Real）",
    "wiki/methods/generative-world-models.md": "生成式世界模型（Generative World Models）",
    "wiki/overview/navigation-slam-autonomy-stack.md": "导航与 SLAM（Navigation and Simultaneous Localization and Mapping, SLAM）",
    "wiki/entities/mujoco.md": "物理引擎（MuJoCo）",
    "wiki/methods/reinforcement-learning.md": "强化学习（Reinforcement Learning, RL）",
    "wiki/queries/real-time-control-middleware-guide.md": "实时运控中间件（Real-Time Control Middleware）",
    "wiki/concepts/ros2-basics.md": "机器人操作系统 2 基础（Robot Operating System 2, ROS 2）",
    "wiki/overview/motor-drive-firmware-bus-protocols.md": "电机驱动器底软通信协议（Motor Drive Firmware Bus Protocols）",
    "wiki/concepts/contact-rich-manipulation.md": "接触丰富型操作（Contact-Rich Manipulation）",
    "wiki/methods/vla.md": "视觉-语言-动作（Vision-Language-Action, VLA）",
    "wiki/concepts/motion-retargeting.md": "动作重定向（Motion Retargeting）",
    "wiki/concepts/whole-body-tracking-pipeline.md": "全身运动跟踪流水线（Whole-Body Tracking Pipeline, WBT）",
    "wiki/entities/humanoid-robot.md": "人形机器人（Humanoid Robot）",
    "wiki/entities/unitree-g1.md": "宇树 G1 人形机器人（Unitree G1）",
    "wiki/methods/behavior-cloning.md": "行为克隆（Behavior Cloning, BC）",
    "wiki/tasks/manipulation.md": "操作（Manipulation）",
    "wiki/tasks/teleoperation.md": "遥操作（Teleoperation）",
    "wiki/tasks/loco-manipulation.md": "移动操作（Loco-Manipulation, Loco-Manip）",
    "wiki/overview/bfm-41-papers-technology-map.md": "行为基础模型技术地图（Behavior Foundation Model, BFM）",
    "wiki/overview/humanoid-motion-cerebellum-technology-map.md": "运动小脑技术地图（Motion Cerebellum）",
    "wiki/overview/humanoid-amp-motion-prior-survey.md": "人形对抗式运动先验（Humanoid Adversarial Motion Prior, AMP）",
    "wiki/overview/humanoid-paper-notebooks-index.md": "人形论文深读笔记（Humanoid Paper Notebooks）",
    "wiki/overview/paper-notebook-category-01-foundational-rl.md": "论文深读 · 基础强化学习（Foundational Reinforcement Learning, RL）",
    "wiki/overview/paper-notebook-category-02-motion-retargeting.md": "论文深读 · 运动重定向（Motion Retargeting）",
    "wiki/overview/paper-notebook-category-03-high-impact-selection.md": "论文深读 · 高影响力精选（High Impact Selection）",
    "wiki/overview/paper-notebook-category-04-loco-manipulation-and-wbc.md": "论文深读 · 运动操作与全身控制（Loco-Manipulation and Whole-Body Control, WBC）",
    "wiki/overview/paper-notebook-category-05-locomotion.md": "论文深读 · 行走运动（Locomotion）",
    "wiki/overview/paper-notebook-category-06-manipulation.md": "论文深读 · 灵巧操作（Manipulation）",
    "wiki/overview/paper-notebook-category-07-teleoperation.md": "论文深读 · 遥操作（Teleoperation）",
    "wiki/overview/paper-notebook-category-08-navigation.md": "论文深读 · 导航（Navigation）",
    "wiki/overview/paper-notebook-category-09-state-estimation.md": "论文深读 · 状态估计（State Estimation）",
    "wiki/overview/paper-notebook-category-10-sim-to-real.md": "论文深读 · 仿真到现实（Simulation to Reality, Sim2Real）",
    "wiki/overview/paper-notebook-category-11-simulation-benchmark.md": "论文深读 · 仿真与基准（Simulation Benchmark）",
    "wiki/overview/paper-notebook-category-12-hardware-design.md": "论文深读 · 硬件设计（Hardware Design）",
    "wiki/overview/paper-notebook-category-13-physics-based-animation.md": "论文深读 · 物理动画（Physics-Based Animation）",
    "wiki/overview/paper-notebook-category-14-human-motion.md": "论文深读 · 人体动作（Human Motion）",
    "wiki/concepts/foundation-policy.md": "基础策略（Foundation Policy）",
    "wiki/overview/multirotor-simulation-planning-control-stack.md": "多旋翼开源栈（Multirotor Stack）",
    "wiki/methods/sonic-motion-tracking.md": "规模化运动跟踪（Supersizing Motion Tracking for Natural Humanoid Control, SONIC）",
    "wiki/overview/humanoid-hardware-101-technology-map.md": "人形硬件技术地图（Humanoid Hardware 101）",
    "wiki/overview/humanoid-actuator-102-technology-map.md": "人形执行器技术地图（Humanoid Actuator 102）",
    "wiki/overview/robot-learning-overview.md": "机器人学习（Robot Learning）",
    "wiki/methods/policy-optimization.md": "策略优化（Policy Optimization）",
    "wiki/methods/amp-reward.md": "对抗运动先验（Adversarial Motion Prior, AMP）",
    "wiki/methods/model-predictive-control.md": "模型预测控制（Model Predictive Control, MPC）",
    "wiki/entities/mimickit.md": "运动模仿与控制（MimicKit）",
    "wiki/entities/isaac-gym-isaac-lab.md": "仿真训练（Isaac Gym / Isaac Lab）",
    "wiki/tasks/humanoid-soccer.md": "人形足球（Humanoid Soccer）",
}
# Paper Notebooks 分类父节点与 wiki 知识页语义等价：社区检测后合并为同一社区，命名取 canonical 枢纽。
# 规范见 schema/naming.md § 图谱社区命名；分类元数据见 schema/paper-notebook-categories.json。
COMMUNITY_HUB_ALIASES: dict[str, str] = {
    "wiki/overview/paper-notebook-category-01-foundational-rl.md": (
        "wiki/methods/reinforcement-learning.md"
    ),
    "wiki/overview/paper-notebook-category-02-motion-retargeting.md": (
        "wiki/concepts/motion-retargeting.md"
    ),
    "wiki/overview/paper-notebook-category-04-loco-manipulation-and-wbc.md": (
        "wiki/tasks/loco-manipulation.md"
    ),
    "wiki/overview/paper-notebook-category-05-locomotion.md": "wiki/tasks/locomotion.md",
    "wiki/overview/paper-notebook-category-06-manipulation.md": "wiki/tasks/manipulation.md",
    "wiki/overview/paper-notebook-category-07-teleoperation.md": "wiki/tasks/teleoperation.md",
    "wiki/overview/paper-notebook-category-08-navigation.md": (
        "wiki/tasks/vision-language-navigation.md"
    ),
    "wiki/overview/paper-notebook-category-09-state-estimation.md": (
        "wiki/concepts/state-estimation.md"
    ),
    "wiki/overview/paper-notebook-category-10-sim-to-real.md": "wiki/concepts/sim2real.md",
    "wiki/overview/paper-notebook-category-12-hardware-design.md": (
        "wiki/overview/humanoid-hardware-101-technology-map.md"
    ),
}


def canonical_community_hub(hub_id: str) -> str:
    """将别名枢纽页解析为 canonical 枢纽（用于社区合并与命名）。"""
    return COMMUNITY_HUB_ALIASES.get(hub_id, hub_id)


def resolve_community_hub_name(hub_id: str, fallback_label: str) -> str:
    """返回社区基名（不含 ` 社区` 后缀）。优先 override，否则回退枢纽页 label。"""
    return COMMUNITY_NAME_OVERRIDES.get(hub_id, fallback_label)


def warn_nonconforming_institution_labels(
    registry: dict[str, dict[str, Any]] | None = None,
) -> None:
    """对未遵循「中文（English）」格式的机构 label 打印 WARNING（不阻塞生成）。"""
    if registry is None:
        registry = INSTITUTION_REGISTRY
    for inst_id, meta in registry.items():
        label = str((meta or {}).get("label", inst_id))
        if INSTITUTION_LABEL_RE.fullmatch(label):
            continue
        print(
            "WARNING: institution label does not match 中文（English） — "
            f"id={inst_id!r} label={label!r}; "
            "update schema/institutions.json (see schema/naming.md)"
        )


def warn_nonconforming_community_hub_names(
    community_meta: dict[str, dict[str, Any]],
) -> None:
    """对未遵循「中文（English）」格式的社区基名打印 WARNING（不阻塞生成）。"""
    for meta in community_meta.values():
        if meta["id"] == OTHER_COMMUNITY_ID:
            continue
        label = str(meta["label"])
        if not label.endswith(" 社区"):
            continue
        hub_name = label[: -len(" 社区")]
        if COMMUNITY_HUB_NAME_RE.fullmatch(hub_name):
            continue
        hub_id = meta.get("hub_id") or "?"
        print(
            "WARNING: community label does not match 中文（English） 社区 — "
            f"hub={hub_id!r} label={label!r}; "
            "add COMMUNITY_NAME_OVERRIDES entry (see schema/naming.md)"
        )


# V22: 当主社区占比超过该阈值时，对其内部做 Louvain 二级拆分。
LARGE_COMMUNITY_SPLIT_RATIO = 0.40
LARGE_COMMUNITY_MIN_SIZE = 30
# resolution > 1.0 偏好更细粒度社区（Reichardt-Bornholdt 形式的 modularity）。
LOUVAIN_RESOLUTION = 1.15
COMMUNITY_WARNING_RATIO = 0.40
# V23: latest_wiki_nodes 默认/上限项数与回看窗口（天）。
LATEST_NODES_DEFAULT = 20
LATEST_NODES_CAP = 30
LATEST_NODES_WINDOW_DAYS = 30
LATEST_NODES_ENV_VAR = "GRAPH_LATEST_NODES_MAX"
# wiki-activity.json：按日导出全部节点（count 与 nodes 长度一致）。
# 更新记录页与热力图筛选依赖全量日志时间线；单日条目过多时由前端折叠展示。
_GIT_LOG_BOUNDARY = "\x01"
_WIKI_ADDED_DATES_CACHE: dict[str, str] | None = None


def _iter_wiki_md_paths() -> list[str]:
    return sorted(
        str(p.relative_to(REPO_ROOT)).replace("\\", "/")
        for p in WIKI_DIR.rglob("*.md")
        if p.is_file()
    )


def wiki_git_added_dates(*, force_refresh: bool = False) -> dict[str, str]:
    """Map ``wiki/...md`` → ISO date (committer) of first git add.

    Mirrors Humanoid_Robot_Learning_Paper_Notebooks ``generate_updates_data.py``:
    scan ``git log --name-status`` newest→oldest with rename aliasing so history
    under a previous path still counts toward the current file.
    """
    global _WIKI_ADDED_DATES_CACHE
    if not force_refresh and _WIKI_ADDED_DATES_CACHE is not None:
        return _WIKI_ADDED_DATES_CACHE

    current_paths = _iter_wiki_md_paths()
    if not current_paths:
        _WIKI_ADDED_DATES_CACHE = {}
        return _WIKI_ADDED_DATES_CACHE

    try:
        result = subprocess.run(
            [
                "git",
                "-C",
                str(REPO_ROOT),
                "-c",
                "core.quotepath=false",
                "log",
                "--topo-order",
                "--no-merges",
                f"--format={_GIT_LOG_BOUNDARY}%cs",
                "--name-status",
                "--",
                "wiki",
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=120,
        )
        if result.returncode != 0:
            _WIKI_ADDED_DATES_CACHE = {}
            return _WIKI_ADDED_DATES_CACHE
    except (subprocess.SubprocessError, OSError):
        _WIKI_ADDED_DATES_CACHE = {}
        return _WIKI_ADDED_DATES_CACHE

    alias = {p: p for p in current_paths}
    added_date: dict[str, str] = {}
    date_str: str | None = None

    for line in result.stdout.splitlines():
        if line.startswith(_GIT_LOG_BOUNDARY):
            date_str = line[1:].strip() or None
            continue
        if not date_str or "\t" not in line:
            continue
        parts = line.split("\t")
        status = parts[0]
        if not status:
            continue
        kind = status[0]

        if kind in ("R", "C") and len(parts) >= 3:
            old, new = parts[1], parts[2]
            cur = alias.get(new)
            if cur is None:
                continue
            if kind == "R":
                if new != old:
                    del alias[new]
                alias[old] = cur
            else:
                added_date[cur] = date_str
                del alias[new]
        elif len(parts) >= 2:
            path = parts[1]
            cur = alias.get(path)
            if cur is None:
                continue
            if kind == "D":
                del alias[path]
                continue
            if kind == "A":
                added_date[cur] = date_str

    _WIKI_ADDED_DATES_CACHE = added_date
    return added_date


def _wiki_node_action(rel: str, log_date: str, added_dates: dict[str, str]) -> str | None:
    """Classify a log-day wiki touch as ``added`` or ``maintained``."""
    first_day = added_dates.get(rel)
    if not first_day:
        return None
    return "added" if first_day == log_date else "maintained"


def wiki_recency_date(content: str, page: Path) -> date:
    """用于「最近更新」排序：取 frontmatter 的 updated / created 与文件 mtime 中的最大值。"""
    candidates: list[date] = []
    if content.startswith("---"):
        end = content.find("\n---", 3)
        if end != -1:
            fm = content[3:end]
            for key in ("updated", "created"):
                match = re.search(rf"^{key}\s*:\s*(\S+)", fm, re.MULTILINE)
                if not match:
                    continue
                raw = match.group(1).strip().strip("'\"")
                try:
                    candidates.append(date.fromisoformat(raw[:10]))
                except ValueError:
                    continue
    try:
        candidates.append(date.fromtimestamp(page.stat().st_mtime))
    except OSError:
        pass
    return max(candidates) if candidates else date.fromtimestamp(0)


def _wiki_node_detail_id(page_id: str) -> str:
    """将 wiki 下的 .md 路径映射为 detail.html 的 id（与 scripts/utils/paths.path_to_id 一致）。"""
    rel = Path(page_id)
    parts = rel.parts
    stem = rel.stem
    if len(parts) >= 2 and parts[0] == "wiki":
        if parts[1] == "entities":
            return f"entity-{stem}"
        return f"wiki-{parts[1]}-{stem}"
    return stem


def _normalize_wiki_rel_from_log_match(raw: str) -> str:
    s = raw.strip().strip("`'\"").rstrip("，。；、）)」』,.;:")
    if "*" in s:
        return s
    if not s.lower().endswith(".md"):
        s = s + ".md"
    return s


def _expand_wiki_glob(pattern: str) -> list[str]:
    """将 log 中的 `wiki/.../*.md` 展开为仓库内存在的相对路径列表。"""
    rel = _normalize_wiki_rel_from_log_match(pattern)
    if "*" not in rel:
        return [rel] if (REPO_ROOT / rel).is_file() else []
    if not rel.lower().endswith(".md"):
        rel = rel + ".md"
    paths: list[str] = []
    for path in REPO_ROOT.glob(rel):
        if path.is_file():
            paths.append(str(path.relative_to(REPO_ROOT)).replace("\\", "/"))
    return sorted(paths)


def _append_latest_node(
    rel: str,
    *,
    node_by_id: dict[str, dict[str, Any]],
    seen: set[str],
    out: list[dict[str, Any]],
    log_date: str,
    added_dates: dict[str, str] | None = None,
) -> None:
    if not rel.startswith("wiki/") or rel in seen or "*" in rel:
        return
    p = REPO_ROOT / rel
    if not p.is_file():
        return
    base = node_by_id.get(rel)
    if not base:
        return
    seen.add(rel)
    entry: dict[str, Any] = {
        "path": rel,
        "detail_id": _wiki_node_detail_id(rel),
        "label": str(base.get("label") or Path(rel).stem),
        "type": str(base.get("type") or ""),
        "recency": log_date,
        "source": "log.md",
    }
    if added_dates is not None:
        action = _wiki_node_action(rel, log_date, added_dates)
        if action:
            entry["action"] = action
    out.append(entry)


def _log_sections(text: str) -> list[str]:
    """按 `## [` 切分 log.md，仅保留以日期标题开头的块，顺序为文件自上而下（新记录在上）。"""
    parts = re.split(r"(?=^## \[)", text, flags=re.MULTILINE)
    out: list[str] = []
    for p in parts:
        p = p.strip()
        if p.startswith("## ["):
            out.append(p)
    return out


def latest_wiki_nodes_from_log(
    nodes: list[dict[str, Any]],
    *,
    max_items: int = LATEST_NODES_DEFAULT,
    window_days: int = LATEST_NODES_WINDOW_DAYS,
) -> list[dict[str, Any]]:
    """从 log.md 解析最近若干日维护日志中出现的 wiki 节点（去重保序）。

    规则：自上而下读取首条 `## [日期] ...` 的日期作为「最新日」；在 ``window_days``
    天内的所有同/早日期日志块中，按出现顺序收集 `wiki/...`（同一路径只保留首次出现），
    且须对应仓库现存文件并在图谱节点中。最终保留前 ``max_items`` 项。不区分 op 类型。
    """
    if max_items <= 0:
        return []
    if not LOG_MD_PATH.is_file():
        return []
    text = LOG_MD_PATH.read_text(encoding="utf-8")
    sections = _log_sections(text)
    if not sections:
        return []
    first_m = re.match(r"^## \[(\d{4}-\d{2}-\d{2})\]", sections[0])
    if not first_m:
        return []
    try:
        target_date = date.fromisoformat(first_m.group(1))
    except ValueError:
        return []
    cutoff_date = target_date - timedelta(days=max(window_days - 1, 0))
    node_by_id: dict[str, dict[str, Any]] = {str(n["id"]): n for n in nodes}
    added_dates = wiki_git_added_dates()
    seen: set[str] = set()
    out: list[dict[str, Any]] = []

    for chunk in sections:
        if len(out) >= max_items:
            break
        date_m = re.match(r"^## \[(\d{4}-\d{2}-\d{2})\]", chunk)
        if not date_m:
            continue
        try:
            chunk_date = date.fromisoformat(date_m.group(1))
        except ValueError:
            continue
        if chunk_date < cutoff_date:
            break
        log_date = date_m.group(1)
        for m in WIKI_GLOB_IN_LOG.finditer(chunk):
            for rel in _expand_wiki_glob(m.group(1)):
                _append_latest_node(
                    rel,
                    node_by_id=node_by_id,
                    seen=seen,
                    out=out,
                    log_date=log_date,
                    added_dates=added_dates,
                )
        for m in WIKI_PATH_IN_LOG.finditer(chunk):
            rel = _normalize_wiki_rel_from_log_match(m.group(0))
            if "*" in rel:
                for expanded in _expand_wiki_glob(rel):
                    _append_latest_node(
                        expanded,
                        node_by_id=node_by_id,
                        seen=seen,
                        out=out,
                        log_date=log_date,
                        added_dates=added_dates,
                    )
                continue
            _append_latest_node(
                rel,
                node_by_id=node_by_id,
                seen=seen,
                out=out,
                log_date=log_date,
                added_dates=added_dates,
            )
    return out[:max_items]


def wiki_activity_from_log(nodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """从 log.md 全量日志汇总每日出现的 wiki 节点（首页热力图按日期筛选用）。

    与 latest_wiki_nodes_from_log 使用同一套路径解析与校验规则，但不限时间
    窗口：同一日期的多个日志块合并、同日去重（跨日可重复出现），仅保留仓库
    现存且在图谱节点中的路径。返回按日期升序的
    ``[{date, count, nodes: [{detail_id, label, type}]}]``，无节点的日期不输出；
    count 为当日节点数，nodes 为当日全部节点（出现顺序）。
    """
    if not LOG_MD_PATH.is_file():
        return []
    sections = _log_sections(LOG_MD_PATH.read_text(encoding="utf-8"))
    node_by_id: dict[str, dict[str, Any]] = {str(n["id"]): n for n in nodes}
    added_dates = wiki_git_added_dates()
    seen_by_date: dict[str, set[str]] = {}
    metas_by_date: dict[str, list[dict[str, Any]]] = {}

    for chunk in sections:
        date_m = re.match(r"^## \[(\d{4}-\d{2}-\d{2})\]", chunk)
        if not date_m:
            continue
        log_date = date_m.group(1)
        try:
            date.fromisoformat(log_date)
        except ValueError:
            continue
        seen = seen_by_date.setdefault(log_date, set())
        day_out = metas_by_date.setdefault(log_date, [])
        for m in WIKI_GLOB_IN_LOG.finditer(chunk):
            for rel in _expand_wiki_glob(m.group(1)):
                _append_latest_node(
                    rel,
                    node_by_id=node_by_id,
                    seen=seen,
                    out=day_out,
                    log_date=log_date,
                    added_dates=added_dates,
                )
        for m in WIKI_PATH_IN_LOG.finditer(chunk):
            rel = _normalize_wiki_rel_from_log_match(m.group(0))
            if "*" in rel:
                for expanded in _expand_wiki_glob(rel):
                    _append_latest_node(
                        expanded,
                        node_by_id=node_by_id,
                        seen=seen,
                        out=day_out,
                        log_date=log_date,
                        added_dates=added_dates,
                    )
                continue
            _append_latest_node(
                rel,
                node_by_id=node_by_id,
                seen=seen,
                out=day_out,
                log_date=log_date,
                added_dates=added_dates,
            )

    days: list[dict[str, Any]] = []
    for log_date in sorted(metas_by_date):
        metas = metas_by_date[log_date]
        if not metas:
            continue
        nodes_out: list[dict[str, Any]] = []
        added_count = 0
        maintained_count = 0
        for meta in metas:
            node_entry: dict[str, Any] = {
                "detail_id": meta["detail_id"],
                "label": meta["label"],
                "type": meta["type"],
            }
            action = meta.get("action")
            if action:
                node_entry["action"] = action
                if action == "added":
                    added_count += 1
                else:
                    maintained_count += 1
            nodes_out.append(node_entry)
        day_entry: dict[str, Any] = {
            "date": log_date,
            "count": len(metas),
            "nodes": nodes_out,
        }
        if added_count:
            day_entry["added_count"] = added_count
        if maintained_count:
            day_entry["maintained_count"] = maintained_count
        days.append(day_entry)
    return days


def resolve_latest_nodes_max(cli_value: int | None) -> int:
    """统一解析 latest_wiki_nodes 上限：CLI > 环境变量 > 默认值，并 clamp 到 [1, CAP]。"""
    candidate: int | None = cli_value
    if candidate is None:
        raw = os.environ.get(LATEST_NODES_ENV_VAR, "").strip()
        if raw:
            try:
                candidate = int(raw)
            except ValueError:
                candidate = None
    if candidate is None:
        return LATEST_NODES_DEFAULT
    return max(1, min(candidate, LATEST_NODES_CAP))


def compute_health_score(content: str) -> int:
    """计算节点健康度（0-3）。

    +1: 有 summary frontmatter
    +1: 有 frontmatter sources 或正文含参考来源区块
    +1: 有 updated frontmatter，或至少包含关联页面区块（说明已纳入交叉引用网络）
    """
    if not content.startswith("---"):
        return 0
    end = content.find("\n---", 3)
    if end == -1:
        return 0
    fm = content[3:end]
    body = content[end + 4 :]
    score = 0
    if re.search(r"^summary\s*:", fm, re.MULTILINE):
        score += 1
    sources_match = re.search(r"^sources\s*:(.*?)(?=^\w|\Z)", fm, re.MULTILINE | re.DOTALL)
    if (sources_match and sources_match.group(1).strip()) or "## 参考来源" in body:
        score += 1
    updated_match = re.search(r"^updated\s*:\s*(\S+)", fm, re.MULTILINE)
    if updated_match:
        try:
            from datetime import date

            updated_date = date.fromisoformat(updated_match.group(1).strip())
            if (date.today() - updated_date).days <= 365:
                score += 1
        except ValueError:
            pass
    elif "## 关联页面" in body:
        score += 1
    return score


def parse_frontmatter_type(content: str) -> str:
    if not content.startswith("---"):
        return ""
    end = content.find("\n---", 3)
    if end == -1:
        return ""
    for line in content[3:end].splitlines():
        if line.strip().startswith("type:"):
            return line.split(":", 1)[1].strip().strip("'\"")
    return ""


def extract_title(content: str) -> str:
    match = re.search(r"^# (.+)", content, re.MULTILINE)
    return match.group(1).strip() if match else ""


def parse_frontmatter_list(content: str, key: str) -> list[str]:
    """提取 frontmatter 某列表字段，支持行内 `key: [a, b]` 与块状 `- item`。"""
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


def derive_node_institutions(content: str, alias_map: dict[str, str] | None = None) -> list[str]:
    """节点「所属机构」（canonical id，去重保序，可多归属）。

    frontmatter 显式 `institutions:` 非空时以其为准（覆盖）；否则从 `tags:` 派生。
    两种来源都经 alias_map 归一到 canonical id，非机构 token 丢弃。
    """
    if alias_map is None:
        alias_map = INSTITUTION_ALIAS_MAP
    explicit = parse_frontmatter_list(content, "institutions")
    source = explicit if explicit else parse_frontmatter_list(content, "tags")
    out: list[str] = []
    seen: set[str] = set()
    for token in source:
        canonical = alias_map.get(str(token).strip().lower())
        if canonical and canonical not in seen:
            seen.add(canonical)
            out.append(canonical)
    return out


def build_institutions_summary(
    nodes: list[dict[str, Any]],
    registry: dict[str, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """汇总各机构命中节点数：[{id, label, size}]，按 size 降序、label 升序。"""
    if registry is None:
        registry = INSTITUTION_REGISTRY
    counts: Counter[str] = Counter()
    for node in nodes:
        for inst_id in node.get("institutions", []) or []:
            counts[inst_id] += 1
    summary = [
        {
            "id": inst_id,
            "label": (registry.get(inst_id) or {}).get("label", inst_id),
            "size": size,
        }
        for inst_id, size in counts.items()
    ]
    summary.sort(key=lambda item: (-int(item["size"]), str(item["label"])))
    return summary


def extract_internal_links(content: str, source_path: Path) -> list[Path]:
    """提取页面中所有指向 wiki/ 目录内部的相对链接。
    支持：
    1. 标准 Markdown: [label](path.md)
    2. Frontmatter related: - path.md
    3. Wikilinks: [[name]]
    """
    targets = []

    def is_wiki_path(p: Path) -> bool:
        try:
            p.relative_to(WIKI_DIR)
            return p.exists()
        except ValueError:
            return False

    # 1. 标准 Markdown 链接
    for match in re.finditer(r"\]\(([^)]+\.md)\)", content):
        href = match.group(1).split("#")[0]
        if href.startswith("http"):
            continue
        resolved = (source_path.parent / href).resolve()
        if is_wiki_path(resolved):
            targets.append(resolved)

    # 2. Frontmatter 'related' 列表
    if content.startswith("---"):
        end = content.find("\n---", 3)
        if end != -1:
            fm = content[3:end]
            related_match = re.search(r"^related\s*:(.*?)(?=^\w|\Z)", fm, re.MULTILINE | re.DOTALL)
            if related_match:
                for line in related_match.group(1).splitlines():
                    line = line.strip().strip("- ")
                    if line.endswith(".md"):
                        resolved = (source_path.parent / line).resolve()
                        if is_wiki_path(resolved):
                            targets.append(resolved)

    # 3. Wikilinks [[name]]
    stem_map = wiki_stem_to_path()
    for match in re.finditer(r"\[\[([^\]|]+)(?:\|[^\]]+)?\]\]", content):
        stem = match.group(1).strip()
        if stem in stem_map:
            targets.append(stem_map[stem])

    return sorted(set(targets), key=lambda path: str(path.relative_to(REPO_ROOT)))


def build_undirected_adjacency(
    node_ids: list[str], edges: list[dict[str, str]]
) -> dict[str, set[str]]:
    adjacency: dict[str, set[str]] = {node_id: set() for node_id in node_ids}
    for edge in edges:
        source = edge["source"]
        target = edge["target"]
        adjacency[source].add(target)
        adjacency[target].add(source)
    return adjacency


def connected_components(adjacency: dict[str, set[str]]) -> list[list[str]]:
    seen: set[str] = set()
    components: list[list[str]] = []
    for start in sorted(adjacency):
        if start in seen:
            continue
        stack = [start]
        component: list[str] = []
        seen.add(start)
        while stack:
            node = stack.pop()
            component.append(node)
            for neighbor in sorted(adjacency[node]):
                if neighbor not in seen:
                    seen.add(neighbor)
                    stack.append(neighbor)
        components.append(sorted(component))
    return sorted(components, key=lambda members: (-len(members), members[0] if members else ""))


def edge_betweenness(adjacency: dict[str, set[str]]) -> dict[tuple[str, str], float]:
    """Brandes edge betweenness for small unweighted graphs."""
    betweenness: dict[tuple[str, str], float] = defaultdict(float)
    for source in sorted(adjacency):
        stack: list[str] = []
        predecessors: dict[str, list[str]] = {node: [] for node in adjacency}
        sigma: dict[str, float] = {node: 0.0 for node in adjacency}
        distance: dict[str, int] = {node: -1 for node in adjacency}
        sigma[source] = 1.0
        distance[source] = 0
        queue = [source]
        head = 0
        while head < len(queue):
            vertex = queue[head]
            head += 1
            stack.append(vertex)
            for neighbor in sorted(adjacency[vertex]):
                if distance[neighbor] < 0:
                    queue.append(neighbor)
                    distance[neighbor] = distance[vertex] + 1
                if distance[neighbor] == distance[vertex] + 1:
                    sigma[neighbor] += sigma[vertex]
                    predecessors[neighbor].append(vertex)

        dependency: dict[str, float] = {node: 0.0 for node in adjacency}
        while stack:
            vertex = stack.pop()
            if sigma[vertex] == 0:
                continue
            for predecessor in predecessors[vertex]:
                contribution = (sigma[predecessor] / sigma[vertex]) * (1.0 + dependency[vertex])
                a, b = predecessor, vertex
                edge: tuple[str, str] = (a, b) if a < b else (b, a)
                betweenness[edge] += contribution
                dependency[predecessor] += contribution

    for edge in list(betweenness):
        betweenness[edge] /= 2.0
    return betweenness


def modularity(partition: list[list[str]], adjacency: dict[str, set[str]]) -> float:
    edge_count = sum(len(neighbors) for neighbors in adjacency.values()) / 2
    if edge_count == 0:
        return 0.0
    degree = {node: len(neighbors) for node, neighbors in adjacency.items()}
    score = 0.0
    for community in partition:
        for i in community:
            for j in community:
                a_ij = 1.0 if j in adjacency[i] else 0.0
                score += a_ij - degree[i] * degree[j] / (2 * edge_count)
    return score / (2 * edge_count)


def _merge_communities_to_cap(
    partition: list[list[str]],
    adjacency: dict[str, set[str]],
    cap: int,
) -> list[list[str]]:
    """将 Louvain 过细分区合并到不超过 cap 个社区（优先合并跨边最少的相邻小社区）。"""
    if len(partition) <= cap:
        return partition

    groups: list[set[str]] = [set(members) for members in partition]
    while len(groups) > cap:
        smallest_idx = min(range(len(groups)), key=lambda i: len(groups[i]))
        small = groups.pop(smallest_idx)
        best_j = 0
        best_cross = -1
        for j, other in enumerate(groups):
            cross = sum(
                1 for node in small for neighbor in adjacency.get(node, ()) if neighbor in other
            )
            if cross > best_cross:
                best_cross = cross
                best_j = j
        groups[best_j].update(small)

    return [sorted(members) for members in groups]


def detect_communities(adjacency: dict[str, set[str]]) -> list[list[str]]:
    """主社区检测：Louvain（O(n log n) 量级）替代 Girvan-Newman 边介数（O(n³)）。"""
    if not adjacency:
        return []

    partition = louvain_communities(adjacency, resolution=LOUVAIN_RESOLUTION)
    if not partition:
        partition = connected_components(adjacency)

    merged = _merge_communities_to_cap(partition, adjacency, PRIMARY_COMMUNITY_CAP)
    refined = refine_oversized_communities(merged, adjacency)
    return sorted(refined, key=lambda members: (-len(members), members[0] if members else ""))


def refine_oversized_communities(
    partition: list[list[str]],
    adjacency: dict[str, set[str]],
) -> list[list[str]]:
    """对超出阈值的巨型社区做 Louvain 二级拆分。

    采用 Reichardt-Bornholdt 带 resolution γ 的 modularity，γ>1 偏好更细粒度社区。
    仅当拆分后子社区个数≥2 且能降低最大社区占比时才采纳。
    """
    total_nodes = sum(len(c) for c in partition)
    if total_nodes == 0:
        return partition

    refined: list[list[str]] = []
    for community in partition:
        ratio = len(community) / total_nodes
        if ratio <= LARGE_COMMUNITY_SPLIT_RATIO or len(community) < LARGE_COMMUNITY_MIN_SIZE:
            refined.append(community)
            continue

        members = set(community)
        sub_adj = {node: adjacency[node] & members for node in community}
        sub_groups = louvain_communities(sub_adj, resolution=LOUVAIN_RESOLUTION)
        if len(sub_groups) >= 2:
            refined.extend(sub_groups)
        else:
            refined.append(community)
    return refined


def louvain_communities(
    adjacency: dict[str, set[str]],
    resolution: float = 1.0,
) -> list[list[str]]:
    """纯 Python Louvain 单层局部移动，无外部依赖。

    模块度增益（无权图）：ΔQ = k_i_in - γ * Σ_tot * k_i / 2m
    """
    nodes = sorted(adjacency.keys())
    if not nodes:
        return []

    total_edges = sum(len(neighbors) for neighbors in adjacency.values()) / 2
    if total_edges == 0:
        return [[node] for node in nodes]

    m2 = 2 * total_edges
    degrees = {node: len(adjacency[node]) for node in nodes}
    node_to_comm = {node: idx for idx, node in enumerate(nodes)}
    comm_degree: dict[int, float] = {}
    for node in nodes:
        comm = node_to_comm[node]
        comm_degree[comm] = comm_degree.get(comm, 0.0) + degrees[node]

    improved = True
    iteration = 0
    max_iterations = 30
    while improved and iteration < max_iterations:
        improved = False
        iteration += 1
        for node in nodes:
            current_comm = node_to_comm[node]
            neighbor_weights: dict[int, int] = {}
            for neighbor in sorted(adjacency[node]):
                nc = node_to_comm[neighbor]
                neighbor_weights[nc] = neighbor_weights.get(nc, 0) + 1

            comm_degree[current_comm] -= degrees[node]
            k_i_in_current = neighbor_weights.get(current_comm, 0)
            best_comm = current_comm
            best_gain = k_i_in_current - resolution * comm_degree[current_comm] * degrees[node] / m2

            for candidate, k_i_in in sorted(neighbor_weights.items(), key=lambda kv: kv[0]):
                if candidate == current_comm:
                    continue
                gain = k_i_in - resolution * comm_degree.get(candidate, 0.0) * degrees[node] / m2
                if gain > best_gain + 1e-12:
                    best_gain = gain
                    best_comm = candidate

            comm_degree[best_comm] = comm_degree.get(best_comm, 0.0) + degrees[node]
            node_to_comm[node] = best_comm
            if best_comm != current_comm:
                improved = True

    groups: dict[int, list[str]] = {}
    for node, comm in node_to_comm.items():
        groups.setdefault(comm, []).append(node)
    return sorted(
        (sorted(members) for members in groups.values()),
        key=lambda members: (-len(members), members[0] if members else ""),
    )


def _hub_for_members(
    members: list[str],
    degree_map: Counter[str],
    node_map: dict[str, dict[str, Any]],
) -> str:
    return max(
        members,
        key=lambda node_id: (degree_map.get(node_id, 0), node_map[node_id]["label"]),
    )


def _merge_partition_by_hub_equivalence(
    partition: list[list[str]],
    degree_map: Counter[str],
    node_map: dict[str, dict[str, Any]],
) -> list[list[str]]:
    """合并枢纽页语义等价的社区分区（如 Paper Notebooks 分类页 vs 对应 task/concept 页）。"""
    if not COMMUNITY_HUB_ALIASES:
        return partition

    buckets: dict[str, set[str]] = defaultdict(set)
    for members in partition:
        hub_id = _hub_for_members(members, degree_map, node_map)
        buckets[canonical_community_hub(hub_id)].update(members)

    merged = [sorted(members) for members in buckets.values()]
    return sorted(merged, key=lambda members: (-len(members), members[0] if members else ""))


def _intra_community_edge_ratio(
    node_id: str,
    community_id: str,
    adjacency: dict[str, set[str]],
    node_to_community: dict[str, str],
) -> float:
    neighbors = adjacency.get(node_id, set())
    if not neighbors:
        return 0.0
    same = sum(1 for nb in neighbors if node_to_community.get(nb) == community_id)
    return same / len(neighbors)


def _community_hub_ids(community_meta: dict[str, dict[str, Any]]) -> set[str]:
    return {
        str(meta["hub_id"])
        for meta in community_meta.values()
        if meta["id"] != OTHER_COMMUNITY_ID and meta.get("hub_id")
    }


def _demote_weak_community_members(
    node_to_community: dict[str, str],
    community_meta: dict[str, dict[str, Any]],
    adjacency: dict[str, set[str]],
    *,
    threshold: float = COMMUNITY_MEMBERSHIP_THRESHOLD,
) -> None:
    """弱归属节点归入「其他社区」：邻居半数以上不在本社区，且非社区枢纽页。"""
    hub_ids = _community_hub_ids(community_meta)
    for node_id, community_id in list(node_to_community.items()):
        if community_id == OTHER_COMMUNITY_ID or node_id in hub_ids:
            continue
        ratio = _intra_community_edge_ratio(node_id, community_id, adjacency, node_to_community)
        if ratio < threshold:
            node_to_community[node_id] = OTHER_COMMUNITY_ID


def _recalculate_community_sizes(
    community_meta: dict[str, dict[str, Any]],
    node_to_community: dict[str, str],
) -> None:
    for meta in community_meta.values():
        meta["size"] = 0
    for community_id in node_to_community.values():
        if community_id in community_meta:
            community_meta[community_id]["size"] += 1


def _ensure_other_community_bucket(community_meta: dict[str, dict[str, Any]]) -> None:
    community_meta.setdefault(
        OTHER_COMMUNITY_ID,
        {"id": OTHER_COMMUNITY_ID, "label": OTHER_COMMUNITY_LABEL, "size": 0, "hub_id": None},
    )


def assign_communities(
    nodes: list[dict[str, Any]],
    edges: list[dict[str, str]],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    node_ids = [node["id"] for node in nodes]
    degree_map: Counter[str] = Counter()
    for edge in edges:
        degree_map[edge["source"]] += 1
        degree_map[edge["target"]] += 1

    adjacency = build_undirected_adjacency(node_ids, edges)
    sorted_groups = _merge_partition_by_hub_equivalence(
        detect_communities(adjacency),
        degree_map,
        {node["id"]: node for node in nodes},
    )

    node_map = {node["id"]: node for node in nodes}
    community_meta: dict[str, dict[str, Any]] = {}
    node_to_community: dict[str, str] = {}

    for idx, members in enumerate(sorted_groups):
        if idx < MAX_COMMUNITIES:
            community_id = f"community-{idx}"
            hub_id = canonical_community_hub(
                _hub_for_members(members, degree_map, node_map),
            )
            hub_name = resolve_community_hub_name(hub_id, node_map[hub_id]["label"])
            label = f"{hub_name} 社区"
        else:
            community_id = OTHER_COMMUNITY_ID
            label = OTHER_COMMUNITY_LABEL
        community_meta.setdefault(
            community_id, {"id": community_id, "label": label, "size": 0, "hub_id": None}
        )
        cm_entry = community_meta[community_id]
        cm_entry["size"] = int(cm_entry["size"]) + len(members)
        if community_meta[community_id]["hub_id"] is None and community_id != OTHER_COMMUNITY_ID:
            community_meta[community_id]["hub_id"] = hub_id
        for node_id in members:
            node_to_community[node_id] = community_id

    _demote_weak_community_members(node_to_community, community_meta, adjacency)
    _ensure_other_community_bucket(community_meta)
    _recalculate_community_sizes(community_meta, node_to_community)

    for node in nodes:
        node["community"] = node_to_community.get(node["id"], OTHER_COMMUNITY_ID)

    community_list = sorted(
        community_meta.values(),
        key=lambda item: (
            item["id"] == OTHER_COMMUNITY_ID,
            -int(item["size"]),
            str(item["label"]),
        ),
    )
    return community_list, community_meta


def _build_graph_data() -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    """扫描所有 wiki 页面，构建节点和边列表。"""
    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, str]] = []
    seen_edges: set[tuple[str, str]] = set()

    for page in sorted(WIKI_DIR.rglob("*.md")):
        if page.name == "README.md":
            continue
        content = page.read_text(encoding="utf-8")
        page_id = str(page.relative_to(REPO_ROOT))
        node_type = parse_frontmatter_type(content)
        node_tags = [str(t).strip().lower() for t in parse_frontmatter_list(content, "tags")]
        node: dict[str, Any] = {
            "id": page_id,
            "label": extract_title(content) or page.stem,
            "type": node_type,
            "health_score": compute_health_score(content),
            "summary": extract_summary(content),
            "_recency": wiki_recency_date(content, page).isoformat(),
            # 论文节点：type=entity 且 frontmatter tags 含 paper（私有标记，写出前剔除）
            "_is_paper": node_type == "entity" and "paper" in node_tags,
        }
        institutions = derive_node_institutions(content)
        if institutions:
            node["institutions"] = institutions
        nodes.append(node)

        for target in extract_internal_links(content, page):
            target_id = str(target.relative_to(REPO_ROOT))
            if page_id == target_id:
                continue
            key = (page_id, target_id)
            if key not in seen_edges:
                seen_edges.add(key)
                edges.append({"source": page_id, "target": target_id})

    return nodes, edges


def _compute_graph_stats(
    nodes: list[dict[str, Any]],
    edges: list[dict[str, str]],
    communities: list[dict[str, Any]],
    community_meta: dict[str, dict[str, Any]],
    *,
    latest_nodes_max: int = LATEST_NODES_DEFAULT,
) -> dict[str, Any]:
    """计算图谱统计数据并写入 graph-stats.json。"""
    in_degree: dict[str, int] = {n["id"]: 0 for n in nodes}
    out_degree: dict[str, int] = {n["id"]: 0 for n in nodes}
    for edge in edges:
        out_degree[edge["source"]] = out_degree.get(edge["source"], 0) + 1
        in_degree[edge["target"]] = in_degree.get(edge["target"], 0) + 1

    total_degree = {
        node["id"]: in_degree.get(node["id"], 0) + out_degree.get(node["id"], 0) for node in nodes
    }

    top_hubs = sorted(nodes, key=lambda node: total_degree.get(node["id"], 0), reverse=True)[:10]
    hub_list = [
        {"id": node["id"], "label": node["label"], "degree": total_degree[node["id"]]}
        for node in top_hubs
    ]

    paper_nodes = [node for node in nodes if node.get("_is_paper")]
    top_paper_hubs = sorted(
        paper_nodes, key=lambda node: total_degree.get(node["id"], 0), reverse=True
    )[:10]
    paper_hub_list = [
        {"id": node["id"], "label": node["label"], "degree": total_degree[node["id"]]}
        for node in top_paper_hubs
    ]

    orphans = [
        {"id": node["id"], "label": node["label"], "out_degree": out_degree.get(node["id"], 0)}
        for node in nodes
        if in_degree.get(node["id"], 0) == 0
    ]

    type_dist: dict[str, int] = {}
    for node in nodes:
        node_type = node.get("type") or "unknown"
        type_dist[node_type] = type_dist.get(node_type, 0) + 1

    community_dist = {
        meta["label"]: int(meta["size"])
        for meta in sorted(community_meta.values(), key=lambda item: -int(item["size"]))
    }

    community_sizes = [
        int(meta["size"]) for meta in community_meta.values() if meta["id"] != OTHER_COMMUNITY_ID
    ]
    singleton_communities = [
        meta["label"]
        for meta in community_meta.values()
        if int(meta["size"]) < 3 and meta["id"] != OTHER_COMMUNITY_ID
    ]
    largest_size = max(community_sizes, default=0)
    largest_ratio = round(largest_size / max(len(nodes), 1), 3)

    latest_wiki_nodes: list[dict[str, Any]] = latest_wiki_nodes_from_log(
        nodes, max_items=latest_nodes_max
    )
    if not latest_wiki_nodes and nodes:
        best = max(
            nodes,
            key=lambda n: (date.fromisoformat(str(n["_recency"])), str(n["id"])),
        )
        latest_wiki_nodes = [
            {
                "path": best["id"],
                "detail_id": _wiki_node_detail_id(best["id"]),
                "label": best["label"],
                "type": best.get("type") or "",
                "recency": best["_recency"],
                "source": "recency",
            }
        ]
    latest_wiki_node: dict[str, Any] | None = latest_wiki_nodes[0] if latest_wiki_nodes else None

    stats = {
        "generated_at": date.today().isoformat(),
        "node_count": len(nodes),
        "edge_count": len(edges),
        "community_count": len(communities),
        "top_hubs": hub_list,
        "top_paper_hubs": paper_hub_list,
        "orphan_nodes": orphans,
        "type_distribution": dict(sorted(type_dist.items(), key=lambda x: x[1], reverse=True)),
        "community_distribution": community_dist,
        "community_quality": {
            "singleton_communities": singleton_communities,
            "largest_community_ratio": largest_ratio,
            "community_quality_warning": largest_ratio > COMMUNITY_WARNING_RATIO,
        },
        "latest_wiki_nodes": latest_wiki_nodes,
        "latest_wiki_node": latest_wiki_node,
    }
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="生成 wiki 内链图谱与统计。")
    parser.add_argument(
        "--latest-nodes-max",
        type=int,
        default=None,
        help=(
            "latest_wiki_nodes 最多保留的节点数 "
            f"（默认 {LATEST_NODES_DEFAULT}，上限 {LATEST_NODES_CAP}；"
            f"亦可通过环境变量 {LATEST_NODES_ENV_VAR} 设置）。"
        ),
    )
    args = parser.parse_args()
    latest_nodes_max = resolve_latest_nodes_max(args.latest_nodes_max)

    nodes, edges = _build_graph_data()
    communities, community_meta = assign_communities(nodes, edges)
    warn_nonconforming_community_hub_names(community_meta)
    warn_nonconforming_institution_labels()

    stats = _compute_graph_stats(
        nodes, edges, communities, community_meta, latest_nodes_max=latest_nodes_max
    )

    for node in nodes:
        node.pop("_is_paper", None)
        recency = node.pop("_recency", None)
        if recency:
            node["recency"] = recency

    institutions = build_institutions_summary(nodes)

    graph = {
        "nodes": nodes,
        "edges": edges,
        "communities": communities,
        "institutions": institutions,
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(
        json.dumps(graph, ensure_ascii=False, separators=(",", ":")), encoding="utf-8"
    )
    print(
        f"✅ link-graph.json: {len(nodes)} nodes, {len(edges)} edges, "
        f"{len(communities)} communities, {len(institutions)} institutions "
        f"→ {OUT_PATH.relative_to(REPO_ROOT)}"
    )

    STATS_PATH.write_text(
        json.dumps(stats, ensure_ascii=False, separators=(",", ":")), encoding="utf-8"
    )
    orphans = stats["orphan_nodes"]
    hub_list = stats["top_hubs"]
    print(
        f"✅ graph-stats.json: {len(orphans)} orphans, "
        f"top hub='{hub_list[0]['label'] if hub_list else '-'}' → {STATS_PATH.relative_to(REPO_ROOT)}"
    )

    activity_days = wiki_activity_from_log(nodes)
    activity = {"generated_at": stats["generated_at"], "days": activity_days}
    ACTIVITY_PATH.write_text(
        json.dumps(activity, ensure_ascii=False, separators=(",", ":")), encoding="utf-8"
    )
    print(
        f"✅ wiki-activity.json: {len(activity_days)} days, "
        f"{sum(d['count'] for d in activity_days)} node refs "
        f"→ {ACTIVITY_PATH.relative_to(REPO_ROOT)}"
    )


if __name__ == "__main__":
    main()
