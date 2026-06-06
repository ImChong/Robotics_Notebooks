#!/usr/bin/env python3
"""
generate_link_graph.py — Wiki 内链图谱生成工具

扫描所有 wiki 页面的内链，生成 exports/link-graph.json，
供 docs/graph.html 的 D3.js 渲染使用。

同时写入 exports/graph-stats.json（含 latest_wiki_nodes：按 log.md 最新日历日
合并当日所有 `## [日期]` 块中出现的有效 wiki/... 路径（去重保序；ingest /
structural / query 等均可）；latest_wiki_node 为当日列表首项（兼容旧字段）。
若无日志命中则回退到 frontmatter / mtime 的 recency，列表仅一项。

输出格式：
  {
    "nodes": [
      {
        "id": "wiki/methods/mpc.md",
        "label": "MPC",
        "type": "method",
        "community": "community-0",
        "community_label": "Reinforcement Learning (RL) 社区"
      }
    ],
    "edges": [{"source": "wiki/methods/mpc.md", "target": "wiki/concepts/wbc.md"}],
    "communities": [{"id": "community-0", "label": "...", "size": 12}]
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
from collections import Counter, defaultdict
from datetime import date, timedelta
from pathlib import Path
from typing import Any

from export_minimal import extract_summary

REPO_ROOT = Path(__file__).parent.parent
WIKI_DIR = REPO_ROOT / "wiki"
OUT_PATH = REPO_ROOT / "exports" / "link-graph.json"
STATS_PATH = REPO_ROOT / "exports" / "graph-stats.json"
LOG_MD_PATH = REPO_ROOT / "log.md"
# log.md 正文中出现的 wiki 相对路径（允许省略 .md，匹配至非标点为止）
WIKI_PATH_IN_LOG = re.compile(r"wiki/(?:[\w./-]+/)+[\w./-]+(?:\.md)?", re.IGNORECASE)
# 反引号内的 wiki 通配路径，如 `wiki/entities/paper-bfm-*.md`
WIKI_GLOB_IN_LOG = re.compile(r"`(wiki/[^`]*\*(?:\.md)?)`", re.IGNORECASE)
# 主社区检测（Girvan-Newman）允许的最大社区数：与历史行为一致。
PRIMARY_COMMUNITY_CAP = 8
# 输出中显式命名的最多社区数：二级拆分后给细分社区更多席位，避免大量节点落入"其他社区"。
MAX_COMMUNITIES = 16
OTHER_COMMUNITY_ID = "community-other"
OTHER_COMMUNITY_LABEL = "其他社区"
# 社区展示名格式：「中文（English） 社区」。规范见 schema/naming.md § 图谱社区命名。
# 社区基名默认取枢纽页 H1，但 H1 风格不一；此处按 hub 路径给出统一 override，脚本再追加 ` 社区`。
# 未命中 override 时回退 H1，并在 generate 阶段对不符合 COMMUNITY_HUB_NAME_RE 的基名打印 WARNING。
COMMUNITY_HUB_NAME_RE = re.compile(
    r"^[\u4e00-\u9fff]"  # 以中文开头
    r"[\u4e00-\u9fff\w\s·/·、，,\-：:]*"  # 中文主名（允许常见标点）
    r"（[^）]+）$"  # 全角括号内的英文/缩写副名
)
COMMUNITY_NAME_OVERRIDES: dict[str, str] = {
    "wiki/overview/humanoid-rl-motion-control-body-system-stack.md": "人形 RL 运动控制（Humanoid RL Locomotion）",
    "wiki/concepts/whole-body-control.md": "全身控制（Whole-Body Control, WBC）",
    "wiki/methods/imitation-learning.md": "模仿学习（Imitation Learning, IL）",
    "wiki/tasks/locomotion.md": "运动控制（Locomotion）",
    "wiki/concepts/sim2real.md": "仿真到现实（Sim2Real）",
    "wiki/methods/generative-world-models.md": "生成式世界模型（Generative World Models）",
    "wiki/overview/navigation-slam-autonomy-stack.md": "导航与 SLAM（Navigation / 自动驾驶）",
    "wiki/entities/mujoco.md": "物理引擎（MuJoCo）",
    "wiki/methods/reinforcement-learning.md": "强化学习（Reinforcement Learning, RL）",
    "wiki/queries/real-time-control-middleware-guide.md": "实时运控中间件（Real-Time Control Middleware）",
    "wiki/concepts/contact-rich-manipulation.md": "接触丰富型操作（Contact-Rich Manipulation）",
    "wiki/methods/vla.md": "视觉-语言-动作（VLA）",
    "wiki/concepts/motion-retargeting.md": "动作重定向（Motion Retargeting）",
    "wiki/entities/humanoid-robot.md": "人形机器人（Humanoid Robot）",
    "wiki/entities/unitree-g1.md": "宇树 G1 人形机器人（Unitree G1）",
    "wiki/methods/behavior-cloning.md": "行为克隆（Behavior Cloning）",
    "wiki/tasks/manipulation.md": "操作（Manipulation）",
    "wiki/tasks/loco-manipulation.md": "移动操作（Loco-Manipulation）",
    "wiki/overview/bfm-41-papers-technology-map.md": "行为基础模型技术地图（BFM）",
    "wiki/concepts/foundation-policy.md": "基础策略（Foundation Policy）",
    "wiki/overview/multirotor-simulation-planning-control-stack.md": "多旋翼开源栈（Multirotor Stack）",
    "wiki/methods/sonic-motion-tracking.md": "规模化运动跟踪（SONIC）",
    "wiki/overview/humanoid-hardware-101-technology-map.md": "人形硬件技术地图（Humanoid Hardware 101）",
    "wiki/overview/robot-learning-overview.md": "机器人学习（Robot Learning）",
    "wiki/methods/policy-optimization.md": "策略优化（Policy Optimization）",
    "wiki/methods/amp-reward.md": "对抗运动先验（AMP & HumanX）",
    "wiki/methods/model-predictive-control.md": "模型预测控制（Model Predictive Control, MPC）",
    "wiki/entities/mimickit.md": "运动模仿与控制（MimicKit）",
}


def resolve_community_hub_name(hub_id: str, fallback_label: str) -> str:
    """返回社区基名（不含 ` 社区` 后缀）。优先 override，否则回退枢纽页 label。"""
    return COMMUNITY_NAME_OVERRIDES.get(hub_id, fallback_label)


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

# Wikilink [[stem]] 解析缓存（wiki 文件集合不变时可复用）
_STEM_TO_PATH: dict[str, Path] | None = None


def _wiki_stem_index() -> dict[str, Path]:
    global _STEM_TO_PATH
    if _STEM_TO_PATH is None:
        _STEM_TO_PATH = {p.stem: p for p in WIKI_DIR.rglob("*.md")}
    return _STEM_TO_PATH


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
    out.append(
        {
            "path": rel,
            "detail_id": _wiki_node_detail_id(rel),
            "label": str(base.get("label") or Path(rel).stem),
            "type": str(base.get("type") or ""),
            "recency": log_date,
            "source": "log.md",
        }
    )


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
                    )
                continue
            _append_latest_node(
                rel,
                node_by_id=node_by_id,
                seen=seen,
                out=out,
                log_date=log_date,
            )
    return out[:max_items]


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
    stem_map = _wiki_stem_index()
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


def detect_communities(adjacency: dict[str, set[str]]) -> list[list[str]]:
    working = {node: set(neighbors) for node, neighbors in adjacency.items()}
    best_partition = connected_components(working)
    best_score = modularity(best_partition, adjacency)

    while sum(len(neighbors) for neighbors in working.values()) > 0:
        betweenness = edge_betweenness(working)
        if not betweenness:
            break
        max_value = max(betweenness.values())
        max_edges = sorted(
            [edge for edge, value in betweenness.items() if value == max_value],
            key=lambda e: e,
        )
        for left, right in max_edges:
            working[left].discard(right)
            working[right].discard(left)
        partition = connected_components(working)
        if len(partition) > PRIMARY_COMMUNITY_CAP:
            break
        score = modularity(partition, adjacency)
        if len(partition) > 1 and score >= best_score:
            best_partition = partition
            best_score = score

    refined = refine_oversized_communities(best_partition, adjacency)
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
    sorted_groups = detect_communities(adjacency)

    node_map = {node["id"]: node for node in nodes}
    community_meta: dict[str, dict[str, Any]] = {}
    node_to_community: dict[str, str] = {}

    for idx, members in enumerate(sorted_groups):
        if idx < MAX_COMMUNITIES:
            community_id = f"community-{idx}"
            hub_id = max(
                members,
                key=lambda node_id: (degree_map.get(node_id, 0), node_map[node_id]["label"]),
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
        nodes.append(
            {
                "id": page_id,
                "label": extract_title(content) or page.stem,
                "type": parse_frontmatter_type(content),
                "health_score": compute_health_score(content),
                "summary": extract_summary(content),
                "_recency": wiki_recency_date(content, page).isoformat(),
            }
        )

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

    stats = _compute_graph_stats(
        nodes, edges, communities, community_meta, latest_nodes_max=latest_nodes_max
    )

    for node in nodes:
        recency = node.pop("_recency", None)
        if recency:
            node["recency"] = recency

    graph = {
        "nodes": nodes,
        "edges": edges,
        "communities": communities,
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(
        json.dumps(graph, ensure_ascii=False, separators=(",", ":")), encoding="utf-8"
    )
    print(
        f"✅ link-graph.json: {len(nodes)} nodes, {len(edges)} edges, "
        f"{len(communities)} communities → {OUT_PATH.relative_to(REPO_ROOT)}"
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


if __name__ == "__main__":
    main()
