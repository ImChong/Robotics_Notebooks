import subprocess
from pathlib import Path

import pytest
import yaml
from check_project_nodes import added_entities, check, paper_ids, resource_url


def page(root: Path, name: str, **meta: object) -> str:
    path = root / "wiki/entities" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("---\n" + yaml.safe_dump(meta) + "---\n\n# Project\n", encoding="utf-8")
    return path.relative_to(root).as_posix()


def test_paper_project_split_without_second_arxiv_field(tmp_path: Path) -> None:
    page(tmp_path, "paper-demo.md", project_id="demo", arxiv="2610.06129")
    page(tmp_path, "project-demo.md", project_id="demo", paper="2610.06129v2")
    errors = check(tmp_path)
    assert any("project_id=demo" in error for error in errors)
    assert any("arxiv=2610.06129" in error for error in errors)


def test_project_url_detects_split_with_different_ids(tmp_path: Path) -> None:
    page(tmp_path, "paper-demo.md", project_id="demo", project="https://lab.github.io/demo/")
    page(
        tmp_path, "site-demo.md", project_id="demo-site", url="http://lab.github.io/demo/index.html"
    )
    assert any("project_url=lab.github.io/demo" in error for error in check(tmp_path))


def test_repeated_id_within_one_unified_page_is_valid(tmp_path: Path) -> None:
    page(tmp_path, "paper-demo.md", arxiv="2610.06129v3", papers=["2610.06129", "2610.06129v1"])
    assert check(tmp_path) == []


def test_distinct_papers_using_same_repo_are_allowed(tmp_path: Path) -> None:
    page(
        tmp_path,
        "paper-v1.md",
        project_id="demo-v1",
        arxiv="2501.01234",
        code="https://github.com/lab/demo",
    )
    new = page(
        tmp_path,
        "paper-v2.md",
        project_id="demo-v2",
        arxiv="2502.01234",
        code="http://github.com/LAB/demo.git/",
    )
    assert check(tmp_path, new_pages={new}) == []


def test_new_source_node_for_existing_project_is_blocked(tmp_path: Path) -> None:
    page(
        tmp_path,
        "paper-demo.md",
        project_id="demo",
        arxiv="2610.06129",
        code="https://github.com/lab/demo",
    )
    new = page(
        tmp_path,
        "repo-demo.md",
        project_id="demo-code",
        code="https://github.com/LAB/demo/tree/main",
    )
    assert any("官方源码" in error for error in check(tmp_path, new_pages={new}))


def test_independent_monorepo_module_requires_documented_scope(tmp_path: Path) -> None:
    page(tmp_path, "paper-demo.md", project_id="demo", code="https://github.com/lab/demo")
    new = page(
        tmp_path,
        "repo-module.md",
        project_id="module",
        code="https://github.com/lab/demo",
        code_scope="tools/module",
    )
    assert check(tmp_path, new_pages={new})
    page(
        tmp_path,
        "repo-module.md",
        project_id="module",
        code="https://github.com/lab/demo",
        code_scope="tools/module",
        project_distinction="独立工具，不是论文部署入口",
    )
    assert check(tmp_path, new_pages={new}) == []


def test_only_new_project_entities_require_identity(tmp_path: Path) -> None:
    page(tmp_path, "paper-legacy.md", arxiv="2501.01234")
    new = page(tmp_path, "repo-new.md", tags=["repo"])
    assert check(tmp_path) == []
    assert any("缺 project_id" in error for error in check(tmp_path, new_pages={new}))


def test_body_references_and_sources_do_not_create_project_nodes(tmp_path: Path) -> None:
    name = page(tmp_path, "paper-demo.md", project_id="demo", arxiv="2610.06129")
    other = page(tmp_path, "paper-other.md", project_id="other", arxiv="2610.01234")
    with (tmp_path / other).open("a") as stream:
        stream.write("[Related paper](https://arxiv.org/abs/2610.06129)\n")
    source = tmp_path / "sources/papers/demo.md"
    source.parent.mkdir(parents=True)
    source.write_text((tmp_path / name).read_text())
    assert check(tmp_path) == []


def test_invalid_identity_and_yaml_fail_closed(tmp_path: Path) -> None:
    page(tmp_path, "paper-demo.md", project_id="", project="not-a-url")
    errors = check(tmp_path)
    assert len(errors) == 2
    (tmp_path / "wiki/entities/bad.md").write_text("---\ncode: [\n---\n")
    assert any("项目元数据无效" in error for error in check(tmp_path))


def test_new_page_cannot_use_placeholder_identity_material(tmp_path: Path) -> None:
    name = page(tmp_path, "paper-demo.md", project_id="demo", arxiv="TODO", code="Coming soon")
    errors = check(tmp_path, new_pages={name})
    assert any("论文编号无效" in error for error in errors)
    assert any("code 必须" in error for error in errors)


def test_normalization_preserves_lab_project_anchors() -> None:
    assert resource_url("https://lab.org/blog?id=one") != resource_url(
        "https://lab.org/blog?id=two"
    )
    assert resource_url("https://lab.org/blog?id=one&utm_source=feed") == resource_url(
        "http://lab.org/blog?id=one"
    )
    assert resource_url("https://lab.org/projects.html#one") != resource_url(
        "https://lab.org/projects.html#two"
    )
    assert (
        resource_url("https://github.com/LAB/Demo.git?tab=readme#code", repository=True)
        == "github.com/lab/demo"
    )
    assert paper_ids({"arxiv": "https://arxiv.org/abs/2606.03441v3", "paper": "2606.03441"}) == {
        "2606.03441"
    }


def test_added_pages_include_untracked_entities_and_reject_bad_base(tmp_path: Path) -> None:
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    old = page(tmp_path, "paper-old.md", arxiv="2501.01234")
    subprocess.run(["git", "add", old], cwd=tmp_path, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.org",
            "commit",
            "-qm",
            "baseline",
        ],
        cwd=tmp_path,
        check=True,
    )
    new = page(tmp_path, "paper-new.md", arxiv="2502.01234")
    assert added_entities(tmp_path, "HEAD") == {new}
    with pytest.raises(subprocess.CalledProcessError):
        added_entities(tmp_path, "missing-base")
