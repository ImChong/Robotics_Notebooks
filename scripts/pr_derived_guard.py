#!/usr/bin/env python3
"""PR 不碰「机器人专属 / 部署生成」文件，从根上避免与 main 合并冲突。

背景：以前每个 PR 都提交 catalog.md、log.md、统计 JSON、README 徽章等全局派生文件，
而 main 上的 export.yml 合并后又重写一遍，并行 PR 必然在同几行冲突。现在：

- ``BOT_OWNED``：仍入库，但只由 main 上的 export.yml 更新（catalog.md 重新生成、
  log.d/ 碎片并入 log.md）。PR 不得修改。
- ``DEPLOY_GENERATED``：已 gitignore，由 pages.yml 部署时生成。PR 不得提交。

用法：
    python3 scripts/pr_derived_guard.py check [--base origin/main]   # CI：PR 改了这些文件就失败
    python3 scripts/pr_derived_guard.py sync                          # 合入 origin/main 并自动消解这些文件的冲突

尚无本脚本的旧分支（本机制上线前开的 PR）可直接用 main 上的版本：
    git fetch origin main && git show origin/main:scripts/pr_derived_guard.py > /tmp/pr_derived_guard.py \
      && python3 /tmp/pr_derived_guard.py sync
"""

from __future__ import annotations

import argparse
import difflib
import subprocess
import sys
import tempfile
from datetime import date
from pathlib import Path


def _repo_root() -> Path:
    """当前工作目录所在仓库根（允许从仓库外的脚本副本运行，见模块说明）。"""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"], check=True, capture_output=True, text=True
        ).stdout.strip()
        return Path(out)
    except (OSError, subprocess.CalledProcessError):
        return Path(__file__).resolve().parent.parent


REPO_ROOT = _repo_root()

BOT_OWNED: tuple[str, ...] = ("catalog.md", "log.md")
DEPLOY_GENERATED: tuple[str, ...] = (
    "exports/graph-stats.json",
    "exports/home-stats.json",
    "exports/lint-report.md",
    "docs/exports/graph-stats.json",
    "docs/exports/home-stats.json",
)
GUARDED = BOT_OWNED + DEPLOY_GENERATED
# 入库文件里曾被统计脚本写数字的行（README 徽章、Hero 数字、SW 缓存版本）。旧分支合入时
# 这些行常冲突：只把「冲突块」按 main 解决，同文件其余改动照常保留。
STAT_STAMPED: tuple[str, ...] = ("README.md", "docs/index.html", "docs/sw.js")


def git(*args: str, check: bool = True) -> str:
    result = subprocess.run(
        ["git", *args], cwd=REPO_ROOT, check=check, capture_output=True, text=True
    )
    return result.stdout


def changed_guarded_paths(base: str) -> list[str]:
    """PR 相对 merge-base 新增/修改（不含删除）的受保护路径。"""
    out = git("diff", "--name-only", "--diff-filter=d", f"{base}...HEAD", "--", *GUARDED)
    return [line for line in out.splitlines() if line.strip()]


def cmd_check(base: str) -> int:
    changed = changed_guarded_paths(base)
    if not changed:
        print("✅ PR 未修改机器人专属 / 部署生成文件")
        return 0
    print("❌ PR 修改了以下文件，它们由 main 上的自动化维护，PR 提交会导致合并冲突：")
    for path in changed:
        print(f"  - {path}")
    print(
        "\n修复：运行 `make sync-main`（合入 origin/main，并把这些文件还原为 main 的版本；"
        "log.md 中新增的条目会自动转存为 log.d/ 碎片），然后提交并推送。\n"
        "日志请用 `make log OP=... DESC=...` 写入 log.d/；catalog.md 与统计文件无需手动生成。"
    )
    return 1


def extract_added_log_text(merge_base: str) -> str:
    """分支相对 merge-base 在 log.md 中新增的文本（按块拼接）。"""
    base = git("show", f"{merge_base}:log.md", check=False).splitlines(keepends=True)
    head = git("show", "HEAD:log.md", check=False).splitlines(keepends=True)
    if not head or base == head:
        return ""
    blocks: list[str] = []
    matcher = difflib.SequenceMatcher(a=base, b=head, autojunk=False)
    for tag, _i1, _i2, j1, j2 in matcher.get_opcodes():
        if tag in ("insert", "replace"):
            blocks.append("".join(head[j1:j2]).strip("\n"))
    return "\n\n".join(b for b in blocks if b.strip())


def unmerged_paths() -> list[str]:
    out = git("diff", "--name-only", "--diff-filter=U")
    return [line for line in out.splitlines() if line.strip()]


def resolve_conflict_hunks_with_theirs(path: str) -> None:
    """三方合并该文件，冲突块取 theirs（main），非冲突改动两边都保留。"""
    with tempfile.TemporaryDirectory() as tmp:
        stages: dict[str, str] = {}
        for stage, name in ((1, "base"), (2, "ours"), (3, "theirs")):
            target = Path(tmp) / name
            target.write_text(git("show", f":{stage}:{path}", check=False), encoding="utf-8")
            stages[name] = str(target)
        subprocess.run(
            ["git", "merge-file", "--theirs", stages["ours"], stages["base"], stages["theirs"]],
            cwd=REPO_ROOT,
            check=True,
        )
        (REPO_ROOT / path).write_text(Path(stages["ours"]).read_text(encoding="utf-8"), "utf-8")
    git("add", "--", path)


def is_tracked(path: str) -> bool:
    return bool(git("ls-files", "--", path).strip())


def cmd_sync(remote: str, branch: str) -> int:
    if git("status", "--porcelain", "--untracked-files=no").strip():
        print("❌ 工作区有未提交的改动，请先 commit 或 stash 后再运行 make sync-main")
        return 1
    ref = f"{remote}/{branch}"
    git("fetch", remote, branch)
    merge_base = git("merge-base", "HEAD", ref).strip()

    # 1. 记下分支写进 log.md 的条目，合并后转存为碎片，避免还原 log.md 时丢失
    added = extract_added_log_text(merge_base)

    # 2. 合入 main 但先不提交（冲突留给下面处理）
    merge = subprocess.run(
        ["git", "merge", "--no-ff", "--no-commit", ref],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    merging = bool(git("rev-parse", "-q", "--verify", "MERGE_HEAD", check=False).strip())
    if merge.returncode != 0 and not merging:
        print(f"❌ git merge 失败：\n{merge.stdout}{merge.stderr}")
        return 1

    frag: Path | None = None
    if added:
        # 合并后再导入：旧分支的 log_md 可能还没有碎片函数，此时磁盘上已是 main 的版本
        sys.path.insert(0, str(REPO_ROOT / "scripts"))
        from log_md import write_log_fragment

        frag = write_log_fragment(
            added + "\n", "migrated", REPO_ROOT / "log.d", today=date.today().isoformat()
        )
        print(f"📝 分支对 log.md 的新增条目已转存为 {frag.relative_to(REPO_ROOT)}")

    # 3. 受保护文件一律以 main 为准 / 移出索引
    for path in BOT_OWNED:
        git("checkout", ref, "--", path)
    for path in DEPLOY_GENERATED:
        if is_tracked(path):
            git("rm", "--cached", "-q", "--", path)
    if frag is not None:
        git("add", str(frag.relative_to(REPO_ROOT)))
    for path in STAT_STAMPED:
        if path in unmerged_paths():
            resolve_conflict_hunks_with_theirs(path)
            print(f"🔧 {path}：统计行冲突按 main 解决，其余改动保留")

    remaining = unmerged_paths()
    if remaining:
        print("❌ 以下文件仍有冲突（非派生文件，需要按内容手工合并后 git add 并 git commit）：")
        for path in remaining:
            print(f"  - {path}")
        return 1

    if merging or git("diff", "--cached", "--name-only").strip():
        msg = f"Merge {ref}（派生文件以 main 为准）" if merging else "fix: 还原机器人专属派生文件"
        git("commit", "-q", "-m", msg)
        print(f"✅ 已提交：{msg}")
    else:
        print("✅ 已与 main 同步，无需提交")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_check = sub.add_parser("check", help="PR 修改了受保护文件则失败")
    p_check.add_argument("--base", default="origin/main")
    p_sync = sub.add_parser("sync", help="合入 main 并自动消解受保护文件冲突")
    p_sync.add_argument("--remote", default="origin")
    p_sync.add_argument("--branch", default="main")
    args = parser.parse_args()
    if args.cmd == "check":
        return cmd_check(args.base)
    return cmd_sync(args.remote, args.branch)


if __name__ == "__main__":
    sys.exit(main())
