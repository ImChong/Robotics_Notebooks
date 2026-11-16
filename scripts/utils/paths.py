from pathlib import Path

from utils.markdown import slugify_ascii


def slugify(value: str) -> str:
    """Convert a string to a URL-friendly slug."""
    return slugify_ascii(value, default="")


def path_to_id(path: Path, repo_root: Path) -> str:
    """Generate a unique ID from a file path."""
    # ⚡ Bolt Optimization: Use path slicing instead of relative_to for speed in hot loops
    parts = path.parts[len(repo_root.parts) :]
    stem = path.stem
    if parts[0] == "wiki":
        if parts[1] == "entities":
            return f"entity-{stem}"
        return f"wiki-{parts[1]}-{stem}"
    if parts[0] == "roadmap":
        return f"roadmap-{stem}"
    if parts[0] == "references":
        return f"reference-{parts[1]}-{stem}"
    if parts[0] == "tech-map":
        if len(parts) >= 3 and parts[1] == "modules":
            return f"tech-node-{parts[2]}-{stem}"
        if len(parts) >= 3 and parts[1] == "research-directions":
            return f"tech-node-research-{stem}"
        return f"tech-node-{stem}"

    # Fallback to a generic slug-based ID
    return slugify("-".join(parts)).removesuffix("-md")
