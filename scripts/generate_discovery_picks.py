#!/usr/bin/env python3
"""Generate daily Robotics Discovery curated picks (docs/discovery-picks.json).

Aggregates public signals from official APIs and structured feeds:
- Hugging Face Daily Papers API
- arXiv cs.RO Atom API
- Upstar Robotics 24h & Weekly rising GitHub repos (Schema.org JSON-LD + optional GitHub metadata)
- Hugging Face Robotics trending Models & Datasets APIs

Deduplicates across papers/repos/models/datasets, matches existing wiki/entities/ nodes,
prefers the latest 24h window (expanding up to 7d when needed), and preserves the last valid
snapshot with a stale notice if all upstream fetches fail.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Literal

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from utils.markdown import parse_frontmatter, slugify_ascii  # noqa: E402
from utils.paths import path_to_id  # noqa: E402

BJT = timezone(timedelta(hours=8))
DEFAULT_OUTPUT = REPO_ROOT / "docs" / "discovery-picks.json"
USER_AGENT = "Robotics-Notebooks-Discovery/1.0 (+https://github.com/ImChong/Robotics_Notebooks)"

ARXIV_ID_RE = re.compile(r"(?<!\d)(\d{4}\.\d{4,5})(?:v\d+)?(?!\d)")
GITHUB_REPO_URL_RE = re.compile(
    r"https?://(?:www\.)?github\.com/([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+?)(?:\.git)?(?=[/?#)\s\"',]|$)",
    re.I,
)
GENERIC_URL_RE = re.compile(r"https?://[^\s<>\"')]+", re.I)
JSON_LD_RE = re.compile(
    r'<script[^>]*type=["\']application/ld\+json["\'][^>]*>(.*?)</script>',
    re.I | re.S,
)
STAR_DELTA_RE = re.compile(
    r"([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)\s+\+(\d+)\s*(?:stars?)?",
    re.I,
)
CJK_RE = re.compile(r"[\u4e00-\u9fff]")
UI_EMOJI_RE = re.compile(r"[\U0001F300-\U0001FAFF\u2600-\u26FF\u2700-\u27BF\uFE0F]")

_YAML_LOADER = getattr(yaml, "CSafeLoader", yaml.SafeLoader)

# Domain topic matchers aligned with Issue #4279 priorities
TOPIC_RULES: list[tuple[str, re.Pattern[str], float]] = [
    ("Humanoid", re.compile(r"\b(humanoid|bipedal?|biped)\b", re.I), 3.0),
    (
        "VLA",
        re.compile(r"\b(vla|vision[- ]language[- ]action|action[- ]chunking)\b", re.I),
        3.2,
    ),
    (
        "WAM",
        re.compile(r"\b(wam|world[- ]action|world model|latent action)\b", re.I),
        2.8,
    ),
    (
        "Embodied AI",
        re.compile(r"\b(embodied|physical ai|agentic robot|robot[- ]use)\b", re.I),
        2.6,
    ),
    (
        "Dexterous",
        re.compile(r"\b(dexterous|bimanual|manipulation|gripper|in-hand|tactile)\b", re.I),
        2.5,
    ),
    (
        "WBC",
        re.compile(r"\b(whole[- ]body|wbc|loco[- ]manipulation|motion tracking)\b", re.I),
        2.5,
    ),
    (
        "RL",
        re.compile(r"\b(reinforcement learning|ppo|rl|self[- ]play|sim[- ]to[- ]real)\b", re.I),
        1.6,
    ),
    (
        "IL",
        re.compile(r"\b(imitation learning|behavior cloning|teleoperat|demonstration)\b", re.I),
        2.0,
    ),
    (
        "Sim2Real",
        re.compile(r"\b(sim2real|sim-to-real|real2sim|isaac|mujoco|simulation)\b", re.I),
        2.2,
    ),
    (
        "SLAM",
        re.compile(r"\b(slam|locali[sz]ation|odometry|navigation|mapping)\b", re.I),
        2.2,
    ),
    (
        "Eval",
        re.compile(r"\b(benchmark|eval(?:s|uation)?|robobench|worldleague|worldevals)\b", re.I),
        1.4,
    ),
]

STRONG_ROBOTICS_RE = re.compile(
    r"\b(robot|robotics|robotic|humanoid|embodied|physical ai|vla|vision[- ]language[- ]action|"
    r"world[- ]action|manipulat|locomot|dexterous|teleoperat|sim2real|sim[- ]to[- ]real|real2sim|"
    r"whole[- ]body|quadruped|bimanual|gripper|tactile|mujoco|isaac ?sim|isaac ?lab|lerobot|"
    r"slam|navigation|mobile robot|drone|uav|actuator|exoskeleton)\b",
    re.I,
)

NEGATIVE_DOMAIN_RE = re.compile(
    r"\b(financial|stock market|portfolio|clinical trial|radiology|pathology|protein folding|"
    r"molecule|cancer|theorem proving|sql|compiler|matrix multiplication|tabular)\b",
    re.I,
)

NON_PROJECT_HOST_PREFIXES = (
    "arxiv.org",
    "huggingface.co",
    "github.com",
    "doi.org",
    "youtu.be",
    "youtube.com",
    "twitter.com",
    "x.com",
)


@dataclass
class KBEntry:
    detail_id: str
    path: str
    title: str
    title_zh: str
    summary_zh: str


@dataclass
class CandidateItem:
    item_id: str
    kind: Literal["paper", "project", "model", "dataset"]
    title_en: str
    title_zh: str = ""
    date: str = ""
    date_type: Literal["published", "updated"] = "published"
    team: str = ""
    contribution_en: str = ""
    contribution_zh: str = ""
    topics: list[str] = field(default_factory=list)
    links: dict[str, str] = field(default_factory=dict)
    signals: list[dict[str, Any]] = field(default_factory=list)
    arxiv_id: str = ""
    github_repo: str = ""
    hf_model_id: str = ""
    hf_dataset_id: str = ""
    project_url: str = ""
    score: float = 0.0
    kb_node: dict[str, str] | None = None
    canonical_keys: set[str] = field(default_factory=set)


FetchTextFn = Callable[[str], str]


def _default_fetch_text(url: str, timeout: int = 20) -> str:
    headers = {"User-Agent": USER_AGENT, "Accept": "application/json, text/html, application/xml"}
    if "api.github.com" in url:
        headers["Accept"] = "application/vnd.github+json"
        token = os.environ.get("GITHUB_TOKEN", "").strip()
        if token:
            headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8", errors="replace")


def normalize_arxiv_id(value: str | None) -> str:
    if not value:
        return ""
    match = ARXIV_ID_RE.search(value.strip())
    return match.group(1) if match else ""


def normalize_github_repo(value: str | None) -> str:
    if not value:
        return ""
    text = value.strip()
    if re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", text):
        owner, repo = text.split("/", 1)
        return f"{owner.lower()}/{repo.lower().removesuffix('.git')}"
    match = GITHUB_REPO_URL_RE.search(text)
    if not match:
        return ""
    owner, repo = match.group(1).split("/", 1)
    if owner.lower() in {"topics", "trending", "orgs", "users", "settings", "features"}:
        return ""
    return f"{owner.lower()}/{repo.lower().removesuffix('.git')}"


def normalize_project_url(value: str | None) -> str:
    if not value:
        return ""
    parsed = urllib.parse.urlsplit(value.strip())
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        return ""
    host = parsed.hostname.lower().removeprefix("www.")
    path = re.sub(r"/index\.html?$", "", parsed.path, flags=re.I).rstrip("/")
    return f"{host}{path.lower()}"


def clean_text_snippet(text: str, max_len: int = 240) -> str:
    cleaned = UI_EMOJI_RE.sub("", text or "")
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if len(cleaned) <= max_len:
        return cleaned
    clipped = cleaned[:max_len]
    for sep in (". ", "。"):
        idx = clipped.rfind(sep)
        if idx >= max_len // 2:
            return clipped[: idx + 1].strip()
    return clipped.rsplit(" ", 1)[0].rstrip(",;:") + "..."


def _as_str_list(val: Any) -> list[str]:
    if isinstance(val, list):
        return [str(v).strip() for v in val if v is not None and str(v).strip()]
    if val is None:
        return []
    text = str(val).strip()
    return [text] if text else []


def _extract_heading_title(body: str, fallback: str) -> tuple[str, str]:
    match = re.search(r"^#\s+(.+)$", body, re.M)
    raw = match.group(1).strip() if match else fallback
    for sep in ("｜", "|", " / "):
        if sep in raw:
            parts = [p.strip() for p in raw.split(sep, 1)]
            cjk_parts = [p for p in parts if CJK_RE.search(p)]
            if cjk_parts:
                return raw, cjk_parts[0]
            return raw, parts[-1] or parts[0]
    return raw, raw if CJK_RE.search(raw) else ""


def build_kb_index(repo_root: Path = REPO_ROOT) -> dict[str, KBEntry]:
    """Build lookup keys from wiki/entities/*.md for deduplicating against existing KB pages."""
    index: dict[str, KBEntry] = {}
    entities_dir = repo_root / "wiki" / "entities"
    if not entities_dir.is_dir():
        return index

    for md_path in sorted(entities_dir.glob("*.md")):
        try:
            raw = md_path.read_text(encoding="utf-8")
        except OSError:
            continue
        fm_block, body = parse_frontmatter(raw)
        if not fm_block:
            continue
        try:
            meta = yaml.load(fm_block, Loader=_YAML_LOADER)
        except yaml.YAMLError:
            continue
        if not isinstance(meta, dict):
            continue

        detail_id = path_to_id(md_path, repo_root)
        rel_path = md_path.relative_to(repo_root).as_posix()
        full_title, title_zh = _extract_heading_title(body, md_path.stem)
        summary_zh = str(meta.get("summary") or "").strip()
        entry = KBEntry(
            detail_id=detail_id,
            path=rel_path,
            title=full_title,
            title_zh=title_zh or full_title,
            summary_zh=summary_zh,
        )

        for key_name in ("arxiv", "paper", "papers"):
            for val in _as_str_list(meta.get(key_name)):
                aid = normalize_arxiv_id(val)
                if aid:
                    index.setdefault(f"arxiv:{aid}", entry)

        for key_name in ("code", "repo", "github", "project", "url", "dataset"):
            for val in _as_str_list(meta.get(key_name)):
                gh = normalize_github_repo(val)
                if gh:
                    index.setdefault(f"repo:{gh}", entry)
                parsed = urllib.parse.urlsplit(val.strip())
                host = (parsed.hostname or "").lower().removeprefix("www.")
                if host == "huggingface.co":
                    parts = [p for p in parsed.path.strip("/").split("/") if p]
                    if len(parts) >= 3 and parts[0] == "datasets":
                        index.setdefault(f"hf_dataset:{parts[1].lower()}/{parts[2].lower()}", entry)
                    elif len(parts) >= 2 and parts[0] not in {"papers", "spaces", "models"}:
                        index.setdefault(f"hf_model:{parts[0].lower()}/{parts[1].lower()}", entry)
                norm = normalize_project_url(val)
                if norm and not norm.startswith(("arxiv.org/", "github.com/")):
                    index.setdefault(f"project:{norm}", entry)
    return index


def extract_topics(text: str, extra_tags: list[str] | None = None) -> tuple[list[str], float]:
    """Score robotics domain relevance and extract concise topic labels."""
    haystack = " ".join([text, *(extra_tags or [])])
    topics: list[str] = []
    score = 0.0
    for label, pattern, weight in TOPIC_RULES:
        if pattern.search(haystack):
            topics.append(label)
            score += weight
    if STRONG_ROBOTICS_RE.search(haystack):
        score += 3.5
    if NEGATIVE_DOMAIN_RE.search(haystack) and not topics:
        score -= 6.0
    return topics[:5], score


def _canonical_keys_for(item: CandidateItem) -> set[str]:
    keys: set[str] = set()
    if item.arxiv_id:
        keys.add(f"arxiv:{item.arxiv_id}")
    if item.github_repo:
        keys.add(f"repo:{item.github_repo.lower()}")
    if item.hf_model_id:
        keys.add(f"hf_model:{item.hf_model_id.lower()}")
    if item.hf_dataset_id:
        keys.add(f"hf_dataset:{item.hf_dataset_id.lower()}")
    if item.project_url:
        norm = normalize_project_url(item.project_url)
        if norm:
            keys.add(f"project:{norm}")
    for link_val in item.links.values():
        gh = normalize_github_repo(link_val)
        if gh:
            keys.add(f"repo:{gh}")
        aid = normalize_arxiv_id(link_val)
        if aid:
            keys.add(f"arxiv:{aid}")
    title_slug = slugify_ascii(item.title_en, default="")
    if title_slug and len(title_slug) >= 6:
        keys.add(f"title:{title_slug}")
    return keys


def fetch_hf_daily_papers(
    today: date,
    fetch_text: FetchTextFn,
    *,
    days: int = 7,
) -> list[CandidateItem]:
    items: list[CandidateItem] = []
    seen_ids: set[str] = set()
    for delta in range(days):
        day_str = (today - timedelta(days=delta)).isoformat()
        raw = fetch_text(f"https://huggingface.co/api/daily_papers?date={day_str}&limit=100")
        payload = json.loads(raw)
        if not isinstance(payload, list):
            continue
        for row in payload:
            if not isinstance(row, dict):
                continue
            raw_paper = row.get("paper")
            paper: dict[str, Any] = raw_paper if isinstance(raw_paper, dict) else {}
            aid = normalize_arxiv_id(str(paper.get("id") or row.get("id") or ""))
            if not aid or aid in seen_ids:
                continue
            title = clean_text_snippet(str(row.get("title") or paper.get("title") or ""), 200)
            summary = str(
                paper.get("ai_summary") or paper.get("summary") or row.get("summary") or ""
            )
            kws = [str(k) for k in (paper.get("ai_keywords") or []) if k]
            topics, rel_score = extract_topics(f"{title} {summary}", kws)
            if rel_score < 3.0:
                continue

            seen_ids.add(aid)
            upvotes = int(paper.get("upvotes") or 0)
            pub_raw = str(row.get("publishedAt") or paper.get("publishedAt") or day_str)
            pub_date = pub_raw[:10] if len(pub_raw) >= 10 else day_str
            raw_org = row.get("organization")
            raw_paper_org = paper.get("organization")
            org: dict[str, Any] = (
                raw_org
                if isinstance(raw_org, dict)
                else (raw_paper_org if isinstance(raw_paper_org, dict) else {})
            )
            team = str(org.get("fullname") or org.get("name") or "").strip()
            if not team:
                authors = paper.get("authors") or []
                names = [
                    str(a.get("name")).strip()
                    for a in authors[:2]
                    if isinstance(a, dict) and a.get("name")
                ]
                team = ", ".join(names) if names else "Community Paper"

            links: dict[str, str] = {"paper": f"https://arxiv.org/abs/{aid}"}
            proj_page = str(paper.get("projectPage") or "").strip()
            gh_url = str(paper.get("githubRepo") or "").strip()
            gh_repo = normalize_github_repo(gh_url)
            if proj_page.startswith(("http://", "https://")) and "github.com/" not in proj_page:
                links["project"] = proj_page
            if gh_repo:
                links["code"] = f"https://github.com/{gh_repo}"
            elif proj_page and normalize_github_repo(proj_page):
                gh_repo = normalize_github_repo(proj_page)
                links["code"] = f"https://github.com/{gh_repo}"

            signals: list[dict[str, Any]] = []
            if upvotes > 0:
                signals.append(
                    {
                        "source_id": "hf-daily-papers",
                        "source_name": "Hugging Face Daily Papers",
                        "label_zh": f"HF Daily Papers {upvotes} 赞",
                        "label_en": f"HF Daily Papers {upvotes} upvotes",
                        "observed_at": today.isoformat(),
                    }
                )
            else:
                signals.append(
                    {
                        "source_id": "hf-daily-papers",
                        "source_name": "Hugging Face Daily Papers",
                        "label_zh": f"收录于 HF Daily Papers ({day_str})",
                        "label_en": f"Featured on HF Daily Papers ({day_str})",
                        "observed_at": today.isoformat(),
                    }
                )
            gh_stars = paper.get("githubStars")
            if isinstance(gh_stars, int) and gh_stars >= 10:
                signals.append(
                    {
                        "source_id": "github",
                        "source_name": "GitHub",
                        "label_zh": f"GitHub {gh_stars} Stars",
                        "label_en": f"GitHub {gh_stars} Stars",
                        "observed_at": today.isoformat(),
                    }
                )

            item = CandidateItem(
                item_id=f"arxiv-{aid}",
                kind="paper",
                title_en=title,
                date=pub_date,
                date_type="published",
                team=team,
                contribution_en=clean_text_snippet(summary, 220),
                topics=topics or ["Embodied AI"],
                links=links,
                signals=signals,
                arxiv_id=aid,
                github_repo=gh_repo,
                project_url=links.get("project", ""),
                score=rel_score + min(upvotes, 250) * 0.08 + (1.2 if gh_repo else 0.0),
            )
            item.canonical_keys = _canonical_keys_for(item)
            items.append(item)
    return items


def parse_arxiv_atom(
    xml_text: str,
    today: date,
    *,
    max_age_days: int = 7,
) -> list[CandidateItem]:
    ns = {"a": "http://www.w3.org/2005/Atom"}
    root = ET.fromstring(xml_text)
    items: list[CandidateItem] = []
    for entry in root.findall("a:entry", ns):
        raw_id = entry.findtext("a:id", "", ns) or ""
        aid = normalize_arxiv_id(raw_id)
        if not aid:
            continue
        title = clean_text_snippet(entry.findtext("a:title", "", ns) or "", 200)
        summary = clean_text_snippet(entry.findtext("a:summary", "", ns) or "", 240)
        pub_raw = (entry.findtext("a:published", "", ns) or "")[:10]
        upd_raw = (entry.findtext("a:updated", "", ns) or "")[:10]
        date_str = pub_raw or upd_raw or today.isoformat()
        date_type: Literal["published", "updated"] = "published" if pub_raw else "updated"
        try:
            parsed_d = date.fromisoformat(date_str)
            if (today - parsed_d).days > max_age_days:
                continue
        except ValueError:
            pass

        authors = [
            (a.findtext("a:name", "", ns) or "").strip() for a in entry.findall("a:author", ns)[:3]
        ]
        authors = [a for a in authors if a]
        team = ", ".join(authors[:2]) + (" et al." if len(authors) > 2 else "")
        if not team:
            team = "arXiv cs.RO Authors"

        full_text = f"{title} {entry.findtext('a:summary', '', ns) or ''}"
        topics, rel_score = extract_topics(full_text)
        if rel_score < 2.5:
            continue

        links: dict[str, str] = {"paper": f"https://arxiv.org/abs/{aid}"}
        gh_repo = ""
        for url in GENERIC_URL_RE.findall(full_text):
            cleaned_url = url.rstrip(".,;:)")
            if not gh_repo:
                gh = normalize_github_repo(cleaned_url)
                if gh:
                    gh_repo = gh
                    links["code"] = f"https://github.com/{gh}"
                    continue
            parsed = urllib.parse.urlsplit(cleaned_url)
            host = (parsed.hostname or "").lower().removeprefix("www.")
            if host and not any(
                host == p or host.endswith("." + p) for p in NON_PROJECT_HOST_PREFIXES
            ):
                links.setdefault("project", cleaned_url)

        age_days = 3
        try:
            age_days = max(0, (today - date.fromisoformat(date_str)).days)
        except ValueError:
            pass
        recency_bonus = max(0.0, 2.5 - age_days * 0.5)

        item = CandidateItem(
            item_id=f"arxiv-{aid}",
            kind="paper",
            title_en=title,
            date=date_str,
            date_type=date_type,
            team=team,
            contribution_en=summary,
            topics=topics or ["Robotics"],
            links=links,
            signals=[
                {
                    "source_id": "arxiv-cs-ro",
                    "source_name": "arXiv Robotics (cs.RO)",
                    "label_zh": f"arXiv cs.RO 最新预印本 ({date_str})",
                    "label_en": f"arXiv cs.RO recent preprint ({date_str})",
                    "observed_at": today.isoformat(),
                }
            ],
            arxiv_id=aid,
            github_repo=gh_repo,
            project_url=links.get("project", ""),
            score=rel_score + recency_bonus + (1.2 if gh_repo else 0.0),
        )
        item.canonical_keys = _canonical_keys_for(item)
        items.append(item)
    return items


def fetch_arxiv_cs_ro(
    today: date,
    fetch_text: FetchTextFn,
    *,
    max_results: int = 40,
) -> list[CandidateItem]:
    url = (
        "https://export.arxiv.org/api/query?"
        f"search_query=cat:cs.RO&sortBy=submittedDate&sortOrder=descending&max_results={max_results}"
    )
    xml_text = fetch_text(url)
    return parse_arxiv_atom(xml_text, today)


def parse_upstar_html(
    html: str,
    period: Literal["day", "week"],
) -> tuple[list[tuple[int, str]], dict[str, int]]:
    """Extract ranked GitHub owner/repo list and optional +N star deltas from Upstar HTML."""
    ranked: list[tuple[int, str]] = []
    for raw_json in JSON_LD_RE.findall(html):
        try:
            obj = json.loads(raw_json)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and obj.get("@type") == "CollectionPage":
            main_entity = obj.get("mainEntity") or {}
            elems = main_entity.get("itemListElement") or []
            for elem in elems:
                if not isinstance(elem, dict):
                    continue
                name = str(elem.get("name") or "").strip()
                pos = int(elem.get("position") or len(ranked) + 1)
                if "/" in name:
                    ranked.append((pos, name))
            if ranked:
                break

    deltas: dict[str, int] = {}
    for slug, delta_str in STAR_DELTA_RE.findall(html):
        deltas.setdefault(slug.lower(), int(delta_str))
    return ranked, deltas


def fetch_upstar_robotics(
    today: date,
    fetch_text: FetchTextFn,
    *,
    period: Literal["day", "week"] = "day",
    enrich_github: bool = True,
    max_repos: int = 7,
) -> list[CandidateItem]:
    url = (
        "https://upstar.dev/t/robotics?p=day"
        if period == "day"
        else "https://upstar.dev/t/robotics"
    )
    source_id = "upstar-robotics-day" if period == "day" else "upstar-robotics-week"
    source_name = (
        "Upstar Robotics 24h Rising" if period == "day" else "Upstar Robotics Weekly Rising"
    )
    html = fetch_text(url)
    ranked, deltas = parse_upstar_html(html, period)
    items: list[CandidateItem] = []
    gh_rate_limited = False

    for pos, repo_slug in ranked[:max_repos]:
        norm_repo = normalize_github_repo(repo_slug)
        if not norm_repo:
            continue
        owner, repo_name = repo_slug.split("/", 1)
        desc = ""
        homepage = ""
        stars = None
        created_date = ""
        pushed_date = ""
        gh_topics: list[str] = []

        if enrich_github and not gh_rate_limited:
            try:
                gh_raw = fetch_text(f"https://api.github.com/repos/{repo_slug}")
                gh_data = json.loads(gh_raw)
                desc = str(gh_data.get("description") or "").strip()
                homepage = str(gh_data.get("homepage") or "").strip()
                stars = gh_data.get("stargazers_count")
                created_date = str(gh_data.get("created_at") or "")[:10]
                pushed_date = str(gh_data.get("pushed_at") or "")[:10]
                gh_topics = [str(t) for t in (gh_data.get("topics") or []) if t]
            except urllib.error.HTTPError as exc:
                if exc.code in (403, 429):
                    gh_rate_limited = True
            except Exception:
                pass

        # Prefer created_at when created in recent 7 days; otherwise pushed_at (updated).
        date_str = today.isoformat()
        date_type: Literal["published", "updated"] = "updated"
        if created_date:
            try:
                if (today - date.fromisoformat(created_date)).days <= 7:
                    date_str = created_date
                    date_type = "published"
                elif pushed_date:
                    date_str = pushed_date
                    date_type = "updated"
            except ValueError:
                pass
        elif pushed_date:
            date_str = pushed_date
            date_type = "updated"

        topics, rel_score = extract_topics(f"{repo_slug} {desc}", gh_topics)
        delta = deltas.get(norm_repo)
        delta_zh = f" · +{delta} Stars" if delta else ""
        delta_en = f" (+{delta} stars)" if delta else ""
        window_zh = "24h 榜" if period == "day" else "周榜"
        window_en = "24h" if period == "day" else "Weekly"

        links: dict[str, str] = {"code": f"https://github.com/{repo_slug}"}
        if homepage.startswith(("http://", "https://")) and "github.com/" not in homepage:
            links["project"] = homepage

        signals = [
            {
                "source_id": source_id,
                "source_name": source_name,
                "label_zh": f"Upstar {window_zh} #{pos}{delta_zh}",
                "label_en": f"Upstar {window_en} #{pos}{delta_en}",
                "observed_at": today.isoformat(),
            }
        ]
        if isinstance(stars, int) and stars > 0:
            signals.append(
                {
                    "source_id": "github",
                    "source_name": "GitHub",
                    "label_zh": f"GitHub {stars} Stars",
                    "label_en": f"GitHub {stars} Stars",
                    "observed_at": today.isoformat(),
                }
            )

        contribution = (
            clean_text_snippet(desc, 220)
            if desc
            else f"Trending open-source robotics repository {repo_slug}."
        )
        item = CandidateItem(
            item_id=f"repo-{slugify_ascii(repo_slug)}",
            kind="project",
            title_en=repo_slug,
            date=date_str,
            date_type=date_type,
            team=owner,
            contribution_en=contribution,
            topics=topics or ["Embodied AI"],
            links=links,
            signals=signals,
            github_repo=norm_repo,
            project_url=links.get("project", ""),
            score=max(rel_score, 4.0)
            + max(0.0, 5.0 - pos * 0.6)
            + (0.6 if period == "day" else 0.2),
        )
        item.canonical_keys = _canonical_keys_for(item)
        items.append(item)
    return items


def fetch_hf_robotics_hub(
    kind: Literal["model", "dataset"],
    today: date,
    fetch_text: FetchTextFn,
    *,
    limit: int = 12,
) -> list[CandidateItem]:
    if kind == "model":
        url = (
            "https://huggingface.co/api/models?"
            f"pipeline_tag=robotics&sort=trendingScore&direction=-1&limit={limit}&full=true"
        )
        source_id = "hf-robotics-models"
        source_name = "Hugging Face Robotics Models"
    else:
        url = (
            "https://huggingface.co/api/datasets?"
            f"filter=task_categories:robotics&sort=trendingScore&direction=-1&limit={limit}&full=true"
        )
        source_id = "hf-robotics-datasets"
        source_name = "Hugging Face Robotics Datasets"

    payload = json.loads(fetch_text(url))
    if not isinstance(payload, list):
        return []

    items: list[CandidateItem] = []
    for pos, row in enumerate(payload, start=1):
        if not isinstance(row, dict):
            continue
        hub_id = str(row.get("id") or "").strip()
        if not hub_id or "/" not in hub_id:
            continue
        author = str(row.get("author") or hub_id.split("/", 1)[0]).strip()
        created = str(row.get("createdAt") or "")[:10]
        modified = str(row.get("lastModified") or "")[:10]

        date_str = modified or created or today.isoformat()
        date_type: Literal["published", "updated"] = "updated"
        if created:
            try:
                if (today - date.fromisoformat(created)).days <= 7:
                    date_str = created
                    date_type = "published"
            except ValueError:
                pass

        # Skip stale entries untouched in > 21 days so picks reflect genuinely recent activity
        try:
            if (today - date.fromisoformat(date_str)).days > 21:
                continue
        except ValueError:
            pass

        tags = [str(t) for t in (row.get("tags") or []) if t]
        aid = ""
        for tag in tags:
            if tag.lower().startswith("arxiv:"):
                aid = normalize_arxiv_id(tag)
                if aid:
                    break

        desc_raw = re.sub(r"\s+", " ", str(row.get("description") or "")).strip()
        desc = clean_text_snippet(desc_raw, 220)
        if not desc:
            kind_word = "robotics model checkpoint" if kind == "model" else "robotics dataset"
            desc = f"Trending open {kind_word} on Hugging Face Hub ({hub_id})."

        topics, rel_score = extract_topics(f"{hub_id} {desc}", tags)
        trending = row.get("trendingScore") or 0
        likes = row.get("likes") or 0

        hub_url = (
            f"https://huggingface.co/{hub_id}"
            if kind == "model"
            else f"https://huggingface.co/datasets/{hub_id}"
        )
        links: dict[str, str] = {kind: hub_url}
        if aid:
            links["paper"] = f"https://arxiv.org/abs/{aid}"

        signals = [
            {
                "source_id": source_id,
                "source_name": source_name,
                "label_zh": f"HF 趋势 #{pos} · {likes} 赞",
                "label_en": f"HF Trending #{pos} · {likes} likes",
                "observed_at": today.isoformat(),
            }
        ]

        item = CandidateItem(
            item_id=f"{kind}-{slugify_ascii(hub_id)}",
            kind=kind,
            title_en=hub_id,
            date=date_str,
            date_type=date_type,
            team=author,
            contribution_en=desc,
            topics=topics or ["Embodied AI"],
            links=links,
            signals=signals,
            arxiv_id=aid,
            hf_model_id=hub_id if kind == "model" else "",
            hf_dataset_id=hub_id if kind == "dataset" else "",
            score=max(rel_score, 3.0)
            + min(float(trending), 30.0) * 0.25
            + max(0.0, 3.0 - pos * 0.4),
        )
        item.canonical_keys = _canonical_keys_for(item)
        items.append(item)
    return items


def merge_and_dedupe(
    candidates: list[CandidateItem],
    kb_index: dict[str, KBEntry],
    existing_by_id: dict[str, dict[str, Any]] | None = None,
) -> list[CandidateItem]:
    """Deduplicate papers, repos, models, and datasets sharing canonical identities."""
    merged: list[CandidateItem] = []
    key_to_idx: dict[str, int] = {}
    existing_by_id = existing_by_id or {}

    kind_priority = {"paper": 4, "project": 3, "model": 2, "dataset": 1}

    for cand in candidates:
        keys = cand.canonical_keys or _canonical_keys_for(cand)
        matched_indices = sorted({key_to_idx[k] for k in keys if k in key_to_idx})

        if not matched_indices:
            idx = len(merged)
            merged.append(cand)
            for k in keys:
                key_to_idx[k] = idx
            continue

        primary_idx = matched_indices[0]
        target = merged[primary_idx]

        # Merge candidate into target
        if kind_priority[cand.kind] > kind_priority[target.kind]:
            target.kind = cand.kind
            target.item_id = cand.item_id
            if len(cand.title_en) > len(target.title_en):
                target.title_en = cand.title_en
        if not target.arxiv_id and cand.arxiv_id:
            target.arxiv_id = cand.arxiv_id
        if not target.github_repo and cand.github_repo:
            target.github_repo = cand.github_repo
        if not target.hf_model_id and cand.hf_model_id:
            target.hf_model_id = cand.hf_model_id
        if not target.hf_dataset_id and cand.hf_dataset_id:
            target.hf_dataset_id = cand.hf_dataset_id
        if not target.project_url and cand.project_url:
            target.project_url = cand.project_url

        # Prefer published date over updated; otherwise prefer newer date
        if cand.date_type == "published" and target.date_type != "published":
            target.date = cand.date
            target.date_type = "published"
        elif cand.date_type == target.date_type and cand.date > target.date:
            target.date = cand.date

        if len(cand.contribution_en) > len(target.contribution_en):
            target.contribution_en = cand.contribution_en
        if target.team in {"", "Community Paper", "arXiv cs.RO Authors"} and cand.team not in {
            "",
            "Community Paper",
            "arXiv cs.RO Authors",
        }:
            target.team = cand.team

        for lk, lv in cand.links.items():
            target.links.setdefault(lk, lv)

        existing_sigs = {(s.get("source_id"), s.get("label_en")) for s in target.signals}
        for sig in cand.signals:
            sig_key = (sig.get("source_id"), sig.get("label_en"))
            if sig_key not in existing_sigs:
                target.signals.append(sig)
                existing_sigs.add(sig_key)

        for topic in cand.topics:
            if topic not in target.topics and len(target.topics) < 5:
                target.topics.append(topic)

        target.score = max(target.score, cand.score) + 1.5
        target.canonical_keys = _canonical_keys_for(target)
        for k in target.canonical_keys:
            key_to_idx[k] = primary_idx

    # Attach KB nodes and preserved translations
    for item in merged:
        item.canonical_keys = _canonical_keys_for(item)
        kb_hit: KBEntry | None = None
        for k in sorted(item.canonical_keys):
            if k in kb_index:
                kb_hit = kb_index[k]
                break
        if kb_hit:
            item.kb_node = {
                "detail_id": kb_hit.detail_id,
                "path": kb_hit.path,
                "url": f"detail.html?id={kb_hit.detail_id}",
                "title": kb_hit.title,
            }
            if not item.title_zh and kb_hit.title_zh:
                item.title_zh = kb_hit.title_zh
            if not item.contribution_zh and kb_hit.summary_zh:
                item.contribution_zh = kb_hit.summary_zh

        prev = existing_by_id.get(item.item_id)
        if prev:
            if prev.get("title_zh"):
                item.title_zh = str(prev["title_zh"])
            if prev.get("contribution_zh"):
                item.contribution_zh = str(prev["contribution_zh"])

        if not item.title_zh:
            item.title_zh = item.title_en
        if not item.contribution_zh:
            item.contribution_zh = item.contribution_en

    return merged


def select_picks(
    items: list[CandidateItem],
    today: date,
    *,
    limit: int = 10,
    window_days: int = 7,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Select balanced top picks prioritizing 24h items and expanding to 7d when needed."""
    bucket_24h: list[CandidateItem] = []
    bucket_7d: list[CandidateItem] = []

    for item in items:
        try:
            age = (today - date.fromisoformat(item.date)).days
        except ValueError:
            age = 3
        if age <= 1:
            bucket_24h.append(item)
        elif age <= window_days:
            bucket_7d.append(item)
        elif item.kind in {"model", "dataset"} and age <= 14:
            bucket_7d.append(item)

    bucket_24h.sort(key=lambda x: (x.score, x.date), reverse=True)
    bucket_7d.sort(key=lambda x: (x.score, x.date), reverse=True)

    ordered_pool: list[tuple[str, CandidateItem]] = [("24h", x) for x in bucket_24h]
    expanded = len(bucket_24h) < limit and len(bucket_7d) > 0
    if expanded:
        ordered_pool.extend(("7d", x) for x in bucket_7d)

    selected: list[tuple[str, CandidateItem]] = []
    selected_ids: set[str] = set()
    counts_by_kind: dict[str, int] = defaultdict(int)

    # First ensure top 24h items are always included (subject to overall kind diversity cap)
    max_per_kind = {"paper": max(5, limit - 4), "project": 3, "model": 2, "dataset": 2}
    for bucket_label, item in ordered_pool:
        if bucket_label != "24h" or len(selected) >= limit:
            break
        if counts_by_kind[item.kind] < max_per_kind.get(item.kind, 3):
            selected.append((bucket_label, item))
            selected_ids.add(item.item_id)
            counts_by_kind[item.kind] += 1

    # Reserve up to 1 strong recent model and 1 strong recent dataset when available
    if len(selected) < limit:
        for reserve_kind in ("model", "dataset"):
            if len(selected) >= limit or counts_by_kind[reserve_kind] >= 1:
                continue
            for bucket_label, item in ordered_pool:
                if item.item_id in selected_ids or item.kind != reserve_kind:
                    continue
                if item.score >= 8.0:
                    selected.append((bucket_label, item))
                    selected_ids.add(item.item_id)
                    counts_by_kind[reserve_kind] += 1
                    break

    deferred: list[tuple[str, CandidateItem]] = []
    for bucket_label, item in ordered_pool:
        if len(selected) >= limit:
            break
        if item.item_id in selected_ids:
            continue
        if counts_by_kind[item.kind] < max_per_kind.get(item.kind, 3):
            selected.append((bucket_label, item))
            selected_ids.add(item.item_id)
            counts_by_kind[item.kind] += 1
        else:
            deferred.append((bucket_label, item))

    for bucket_label, item in deferred:
        if len(selected) >= limit:
            break
        if item.item_id not in selected_ids:
            selected.append((bucket_label, item))
            selected_ids.add(item.item_id)

    serialized: list[dict[str, Any]] = []
    for bucket_label, item in selected:
        record: dict[str, Any] = {
            "id": item.item_id,
            "kind": item.kind,
            "title_zh": item.title_zh or item.title_en,
            "title_en": item.title_en,
            "date": item.date,
            "date_type": item.date_type,
            "window_bucket": bucket_label,
            "team": item.team,
            "contribution_zh": item.contribution_zh or item.contribution_en,
            "contribution_en": item.contribution_en,
            "topics": item.topics,
            "links": item.links,
            "signals": item.signals,
        }
        if item.kb_node:
            record["kb_node"] = item.kb_node
        serialized.append(record)

    count_24h = sum(1 for b, _ in selected if b == "24h")
    count_7d = sum(1 for b, _ in selected if b == "7d")
    if count_7d > 0:
        note_zh = (
            f"优先展示最近 24 小时精选（{count_24h} 项），并按近期热度与方向代表性"
            f"扩展至近 {window_days} 天（补充 {count_7d} 项）"
        )
        note_en = (
            f"Prioritizes picks from the past 24h ({count_24h}), expanded to the last "
            f"{window_days} days (+{count_7d}) for topic coverage"
        )
    else:
        note_zh = f"全部精选均来自最近 24 小时公开更新（共 {count_24h} 项）"
        note_en = f"All {count_24h} picks are from the past 24 hours"

    window_meta = {
        "primary_hours": 24,
        "expanded": count_7d > 0,
        "max_days": window_days,
        "count_24h": count_24h,
        "count_7d": count_7d,
        "note_zh": note_zh,
        "note_en": note_en,
    }
    return serialized, window_meta


def load_existing_picks(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None
    except (OSError, json.JSONDecodeError):
        return None


def generate_picks_payload(
    *,
    repo_root: Path = REPO_ROOT,
    output_path: Path = DEFAULT_OUTPUT,
    limit: int = 10,
    window_days: int = 7,
    today: date | None = None,
    now_bjt: datetime | None = None,
    fetch_text: FetchTextFn = _default_fetch_text,
    preserve_translations: bool = True,
    enrich_github: bool = True,
) -> dict[str, Any]:
    now_dt = now_bjt or datetime.now(BJT)
    today_date = today or now_dt.date()
    attempt_iso = now_dt.isoformat(timespec="seconds")

    existing = load_existing_picks(output_path)
    existing_by_id: dict[str, dict[str, Any]] = {}
    if preserve_translations and existing and isinstance(existing.get("items"), list):
        for entry in existing["items"]:
            if isinstance(entry, dict) and entry.get("id"):
                existing_by_id[str(entry["id"])] = entry

    kb_index = build_kb_index(repo_root)
    candidates: list[CandidateItem] = []
    sources_checked = [
        "hf-daily-papers",
        "arxiv-cs-ro",
        "upstar-robotics-day",
        "upstar-robotics-week",
        "hf-robotics-models",
        "hf-robotics-datasets",
    ]
    sources_failed: list[str] = []

    fetchers: list[tuple[str, Callable[[], list[CandidateItem]]]] = [
        (
            "hf-daily-papers",
            lambda: fetch_hf_daily_papers(today_date, fetch_text, days=window_days),
        ),
        ("arxiv-cs-ro", lambda: fetch_arxiv_cs_ro(today_date, fetch_text)),
        (
            "upstar-robotics-day",
            lambda: fetch_upstar_robotics(
                today_date, fetch_text, period="day", enrich_github=enrich_github
            ),
        ),
        (
            "upstar-robotics-week",
            lambda: fetch_upstar_robotics(
                today_date, fetch_text, period="week", enrich_github=enrich_github
            ),
        ),
        (
            "hf-robotics-models",
            lambda: fetch_hf_robotics_hub("model", today_date, fetch_text),
        ),
        (
            "hf-robotics-datasets",
            lambda: fetch_hf_robotics_hub("dataset", today_date, fetch_text),
        ),
    ]

    for src_id, fn in fetchers:
        try:
            candidates.extend(fn())
        except Exception as exc:
            print(f"[discovery] warn: source {src_id} failed: {exc}", file=sys.stderr)
            sources_failed.append(src_id)

    merged = merge_and_dedupe(candidates, kb_index, existing_by_id=existing_by_id)
    selected_items, window_meta = select_picks(
        merged, today_date, limit=limit, window_days=window_days
    )

    if not selected_items:
        # Preserve previous valid snapshot if all upstream sources failed
        if existing and isinstance(existing.get("items"), list) and existing["items"]:
            preserved = dict(existing)
            preserved["status"] = {
                "state": "stale",
                "last_attempt_at": attempt_iso,
                "sources_checked": sources_checked,
                "sources_failed": sources_failed,
                "message_zh": "本次精选更新失败或无可用数据，已保留上次有效精选内容（上方推荐网站导航不受影响）。",
                "message_en": "Latest refresh failed or returned no items; showing the last valid snapshot (website directory remains available).",
            }
            return preserved
        return {
            "schema_version": 1,
            "updated_at": attempt_iso,
            "updated_date_bjt": today_date.isoformat(),
            "status": {
                "state": "stale",
                "last_attempt_at": attempt_iso,
                "sources_checked": sources_checked,
                "sources_failed": sources_failed,
                "message_zh": "暂未获取到最新精选数据，请直接访问上方推荐网站浏览原站榜单。",
                "message_en": "Unable to fetch recent picks; please visit the recommended websites above directly.",
            },
            "window": window_meta,
            "items": [],
        }

    state: Literal["ok", "partial"] = "partial" if sources_failed else "ok"
    if state == "partial":
        msg_zh = (
            f"精选更新完成（部分源暂不可用：{', '.join(sources_failed)}），其余公开来源已正常聚合。"
        )
        msg_en = (
            f"Refreshed with partial sources (unavailable: {', '.join(sources_failed)}); "
            "other public feeds aggregated normally."
        )
    else:
        msg_zh = "已从公开 API 与结构化榜单完成核验、去重与知识库节点匹配。"
        msg_en = "Verified, deduplicated, and matched against knowledge base nodes."

    return {
        "schema_version": 1,
        "updated_at": attempt_iso,
        "updated_date_bjt": today_date.isoformat(),
        "status": {
            "state": state,
            "last_attempt_at": attempt_iso,
            "sources_checked": sources_checked,
            "sources_failed": sources_failed,
            "message_zh": msg_zh,
            "message_en": msg_en,
        },
        "window": window_meta,
        "items": selected_items,
    }


def write_picks_payload(payload: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch and curate recent robotics papers, repos, models, and datasets."
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--window-days", type=int, default=7)
    parser.add_argument("--today", type=str, default="", help="Override YYYY-MM-DD (BJT) for tests")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--no-preserve-translations",
        action="store_true",
        help="Do not reuse existing title_zh/contribution_zh from output file",
    )
    parser.add_argument(
        "--no-github-enrich",
        action="store_true",
        help="Skip optional api.github.com repo enrichment calls",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    now_dt = datetime.now(BJT)
    today_date = date.fromisoformat(args.today) if args.today else now_dt.date()
    payload = generate_picks_payload(
        repo_root=REPO_ROOT,
        output_path=args.output,
        limit=max(1, args.limit),
        window_days=max(1, args.window_days),
        today=today_date,
        now_bjt=now_dt,
        preserve_translations=not args.no_preserve_translations,
        enrich_github=not args.no_github_enrich,
    )
    if args.dry_run:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        write_picks_payload(payload, args.output)
        print(
            f"[discovery] wrote {len(payload.get('items', []))} items to {args.output} "
            f"(state={payload.get('status', {}).get('state')})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
