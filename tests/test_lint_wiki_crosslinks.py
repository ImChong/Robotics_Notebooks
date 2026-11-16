"""Parametrized crosslink lint checks across all six vertical hub rules."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import lint_wiki as lw
import pytest

CROSSLINK_SPECS: list[
    tuple[
        str,
        Callable[[list[Path], dict[str, Any]], None],
        tuple[str, ...],
        str,
        str,
        str,
        str,
        str,
        str,
        str,
        str,
    ]
] = [
    (
        "physics_concept_crosslink",
        lw._check_physics_concept_crosslink,
        ("concepts", "formalizations"),
        "concepts",
        "sim2real-contact.md",
        "[dynamics, contact]",
        "../queries/simulation-physics-fidelity.md",
        "concepts/centroidal-dynamics.md",
        "[dynamics, locomotion, mpc]",
        "formalizations/articulated-body-algorithms.md",
        "simulation-physics-fidelity.md",
    ),
    (
        "contact_control_crosslink",
        lw._check_contact_control_crosslink,
        ("concepts",),
        "concepts",
        "variable-impedance.md",
        "[control, impedance-control, force-control]",
        "../queries/contact-wrench-closed-loop.md",
        "concepts/contact-rich-manipulation.md",
        "[manipulation, contact, force-control]",
        "concepts/tactile-sensing.md",
        "contact-wrench-closed-loop.md",
    ),
    (
        "embodied_fm_crosslink",
        lw._check_embodied_fm_crosslink,
        ("concepts", "comparisons"),
        "concepts",
        "embodied-fm-latency-generalization-tradeoff.md",
        "[vla, world-model, control]",
        "../queries/embodied-fm-taxonomy-loop.md",
        "concepts/world-action-models.md",
        "[world-model, vla, generative]",
        "comparisons/vlm-vln-vla-vlx-world-model-taxonomy.md",
        "embodied-fm-taxonomy-loop.md",
    ),
    (
        "eval_benchmark_crosslink",
        lw._check_eval_benchmark_crosslink,
        ("entities", "comparisons", "concepts"),
        "entities",
        "robo-bench.md",
        "[benchmark, mllm, evaluation]",
        "../queries/embodied-eval-benchmark-selection-loop.md",
        "entities/ewmbench.md",
        "[benchmark, world-model, evaluation]",
        "comparisons/embodied-benchmark-family-compare.md",
        "embodied-eval-benchmark-selection-loop.md",
    ),
    (
        "actuator_drive_chain_crosslink",
        lw._check_actuator_drive_chain_crosslink,
        ("entities", "comparisons", "concepts"),
        "entities",
        "simplefoc.md",
        "[foc, driver, open-source]",
        "../queries/actuator-drive-chain-selection-loop.md",
        "entities/kicad.md",
        "[eda, pcb, open-source]",
        "comparisons/kicad-vs-altium.md",
        "actuator-drive-chain-selection-loop.md",
    ),
    (
        "perception_stack_crosslink",
        lw._check_perception_stack_crosslink,
        ("entities", "comparisons", "concepts", "methods"),
        "entities",
        "ultralytics.md",
        "[tooling, object-detection, perception]",
        "../queries/robot-perception-stack-selection-loop.md",
        "entities/rf-detr.md",
        "[object-detection, instance-segmentation, real-time]",
        "comparisons/yolo-vs-detr.md",
        "robot-perception-stack-selection-loop.md",
    ),
]


def _setup_wiki(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, dirs: tuple[str, ...]) -> Path:
    monkeypatch.setattr(lw, "REPO_ROOT", tmp_path)
    wiki = tmp_path / "wiki"
    for d in set(dirs) | {"concepts", "entities"}:
        (wiki / d).mkdir(parents=True, exist_ok=True)
    return wiki


def _run(
    checker: Callable[[list[Path], dict[str, Any]], None], pages: list[Path]
) -> dict[str, Any]:
    results = lw._empty_results()
    checker(pages, results)
    return results


@pytest.mark.parametrize(
    "key,checker,dirs,pass_dir,pass_name,pass_tags,hub_link,flag1_rel,flag1_tags,flag2_rel,hub_self_name",
    CROSSLINK_SPECS,
    ids=[spec[0] for spec in CROSSLINK_SPECS],
)
def test_crosslink_core_scenarios(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    key: str,
    checker: Callable[[list[Path], dict[str, Any]], None],
    dirs: tuple[str, ...],
    pass_dir: str,
    pass_name: str,
    pass_tags: str,
    hub_link: str,
    flag1_rel: str,
    flag1_tags: str,
    flag2_rel: str,
    hub_self_name: str,
) -> None:
    wiki = _setup_wiki(tmp_path, monkeypatch, dirs)

    # 1. With hub backlink -> passes
    p_ok = wiki / pass_dir / pass_name
    p_ok.write_text(
        f"---\ntype: {pass_dir.rstrip('s')}\ntags: {pass_tags}\n---\n见 [枢纽]({hub_link})。\n",
        encoding="utf-8",
    )
    assert _run(checker, [p_ok])[key] == []

    # 2. Inline tag without hub -> flagged
    p_bad1 = wiki / flag1_rel
    p_bad1.write_text(
        f"---\ntype: concept\ntags: {flag1_tags}\n---\n未回链枢纽。\n",
        encoding="utf-8",
    )
    assert _run(checker, [p_bad1])[key] == [f"wiki/{flag1_rel}"]

    # 3. Secondary page without hub -> flagged
    p_bad2 = wiki / flag2_rel
    p_bad2.write_text(
        f"---\ntype: comparison\ntags: {flag1_tags}\n---\n未回链枢纽。\n",
        encoding="utf-8",
    )
    assert _run(checker, [p_bad2])[key] == [f"wiki/{flag2_rel}"]

    # 4. Unrelated tags -> ignored
    p_ign = wiki / pass_dir / "diffusion-policy.md"
    p_ign.write_text(
        "---\ntype: concept\ntags: [imitation-learning, generative]\n---\n无关内容。\n",
        encoding="utf-8",
    )
    assert _run(checker, [p_ign])[key] == []

    # 5. Hub page itself exempt from self-backlink
    p_hub = wiki / "concepts" / hub_self_name
    p_hub.write_text(
        f"---\ntype: concept\ntags: {pass_tags}\n---\n枢纽页自身无需自链。\n",
        encoding="utf-8",
    )
    assert _run(checker, [p_hub])[key] == []

    # 6. Info-only key does not contribute to failing total
    res = lw._empty_results()
    res[key].append(f"wiki/{flag1_rel}")
    assert lw._failing_total(res) == 0
    assert lw._info_total(res) == 1


@pytest.mark.parametrize(
    "key,checker,subpath,tags,hub_rel",
    [
        (
            "contact_control_crosslink",
            lw._check_contact_control_crosslink,
            "concepts/impedance.md",
            "[control, manipulation, force-control]",
            "../overview/hub-contact-force-control.md",
        ),
        (
            "embodied_fm_crosslink",
            lw._check_embodied_fm_crosslink,
            "concepts/unified-multimodal-tokens.md",
            "[vlx, multimodal]",
            "../overview/hub-embodied-foundation-model.md",
        ),
        (
            "eval_benchmark_crosslink",
            lw._check_eval_benchmark_crosslink,
            "entities/esi-bench.md",
            "[benchmark, evaluation]",
            "../overview/hub-embodied-eval-benchmark.md",
        ),
        (
            "actuator_drive_chain_crosslink",
            lw._check_actuator_drive_chain_crosslink,
            "entities/altium-designer.md",
            "[eda, pcb]",
            "../overview/hub-actuator-drive-chain.md",
        ),
        (
            "perception_stack_crosslink",
            lw._check_perception_stack_crosslink,
            "entities/paper-segment-anything.md",
            "[foundation-model, segmentation]",
            "../overview/hub-perception-stack.md",
        ),
    ],
)
def test_topic_hub_backlink_also_passes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    key: str,
    checker: Callable[[list[Path], dict[str, Any]], None],
    subpath: str,
    tags: str,
    hub_rel: str,
) -> None:
    wiki = _setup_wiki(tmp_path, monkeypatch, ("concepts", "entities"))
    page = wiki / subpath
    page.write_text(
        f"---\ntype: concept\ntags: {tags}\n---\n见 [纵深]({hub_rel})。\n",
        encoding="utf-8",
    )
    assert _run(checker, [page])[key] == []


@pytest.mark.parametrize(
    "key,checker,subpath,list_tags",
    [
        (
            "physics_concept_crosslink",
            lw._check_physics_concept_crosslink,
            "formalizations/aba.md",
            ["dynamics", "aba"],
        ),
        (
            "contact_control_crosslink",
            lw._check_contact_control_crosslink,
            "concepts/tactile.md",
            ["perception", "tactile-sensing"],
        ),
        (
            "embodied_fm_crosslink",
            lw._check_embodied_fm_crosslink,
            "concepts/vln.md",
            ["navigation", "vln-policy"],
        ),
        (
            "eval_benchmark_crosslink",
            lw._check_eval_benchmark_crosslink,
            "entities/paper-eval.md",
            ["policy", "policy-evaluation"],
        ),
        (
            "actuator_drive_chain_crosslink",
            lw._check_actuator_drive_chain_crosslink,
            "entities/bam.md",
            ["modeling", "actuator-network"],
        ),
        (
            "perception_stack_crosslink",
            lw._check_perception_stack_crosslink,
            "entities/findanything.md",
            ["project", "semantic-mapping"],
        ),
    ],
)
def test_list_style_tags_flagged(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    key: str,
    checker: Callable[[list[Path], dict[str, Any]], None],
    subpath: str,
    list_tags: list[str],
) -> None:
    wiki = _setup_wiki(tmp_path, monkeypatch, ("concepts", "entities", "formalizations"))
    page = wiki / subpath
    tag_yaml = "\n".join(f"  - {t}" for t in list_tags)
    page.write_text(f"---\ntype: concept\ntags:\n{tag_yaml}\n---\n未回链。\n", encoding="utf-8")
    assert _run(checker, [page])[key] == [f"wiki/{subpath}"]


def test_eval_benchmark_science_vs_robotics_domain(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    wiki = _setup_wiki(tmp_path, monkeypatch, ("entities",))
    science = wiki / "entities" / "wet-lab-prediction.md"
    science.write_text(
        "---\ntype: entity\ntags: [benchmark, ai-for-science, wet-lab, ai-agent-evaluation]\n"
        "---\n[AI Agent 评测](../concepts/ai-agent-evaluation.md)\n",
        encoding="utf-8",
    )
    robotics = wiki / "entities" / "robotic-lab.md"
    robotics.write_text(
        "---\ntype: entity\ntags:\n  - benchmark\n  - ai-for-science\n"
        "  - robotic-lab\n---\n实验室机器人任务评测，尚缺具身枢纽。\n",
        encoding="utf-8",
    )
    results = _run(lw._check_eval_benchmark_crosslink, [science, robotics])
    assert results["eval_benchmark_crosslink"] == ["wiki/entities/robotic-lab.md"]


def test_actuator_substring_lookalike_and_focal_loss_exempt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    wiki = _setup_wiki(tmp_path, monkeypatch, ("concepts", "entities"))
    lookalikes = {
        "concepts/impedance-control.md": "[impedance-control, force-control]",
        "concepts/lip-zmp.md": "[locomotion, bipedal]",
        "concepts/humanoid-rubber-man-analogy.md": "[humanoid, pedagogy]",
        "entities/paper-xrobotoolkit.md": "[teleop, bytedance]",
        "entities/retinanet.md": "[object-detection, focal-loss, one-stage]",
    }
    pages = []
    for rel, tags in lookalikes.items():
        page = wiki / rel
        page.write_text(f"---\ntype: concept\ntags: {tags}\n---\n未回链。\n", encoding="utf-8")
        pages.append(page)
    results = _run(lw._check_actuator_drive_chain_crosslink, pages)
    assert results["actuator_drive_chain_crosslink"] == []

    p1 = wiki / "concepts" / "field-oriented-control.md"
    p1.write_text("---\ntype: concept\ntags: [actuators, motor]\n---\n未回链。\n", encoding="utf-8")
    p2 = wiki / "entities" / "some-driver.md"
    p2.write_text("---\ntype: entity\ntags: [foc-driver]\n---\n未回链。\n", encoding="utf-8")
    flagged = _run(lw._check_actuator_drive_chain_crosslink, [p1, p2])
    assert sorted(flagged["actuator_drive_chain_crosslink"]) == [
        "wiki/concepts/field-oriented-control.md",
        "wiki/entities/some-driver.md",
    ]


def test_perception_stack_method_and_lookalike_tokens(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    wiki = _setup_wiki(tmp_path, monkeypatch, ("concepts", "entities", "methods"))
    m_page = wiki / "methods" / "object-detection.md"
    m_page.write_text(
        "---\ntype: method\ntags: [object-detection, computer-vision]\n---\n未回链。\n",
        encoding="utf-8",
    )
    assert _run(lw._check_perception_stack_crosslink, [m_page])["perception_stack_crosslink"] == [
        "wiki/methods/object-detection.md"
    ]

    p_ign = wiki / "concepts" / "imitation-learning.md"
    p_ign.write_text(
        "---\ntype: concept\ntags: [imitation-learning, behavior-cloning]\n---\n正文。\n",
        encoding="utf-8",
    )
    p_rec = wiki / "concepts" / "reception-desk.md"
    p_rec.write_text("---\ntype: concept\ntags: [reception, ui]\n---\n正文。\n", encoding="utf-8")
    assert (
        _run(lw._check_perception_stack_crosslink, [p_ign, p_rec])["perception_stack_crosslink"]
        == []
    )

    p_sam = wiki / "entities" / "sam.md"
    p_sam.write_text(
        "---\ntype: entity\ntags: [promptable-segmentation, sam]\n---\n正文。\n", encoding="utf-8"
    )
    p_det = wiki / "entities" / "some-detector.md"
    p_det.write_text("---\ntype: entity\ntags: [detections]\n---\n正文。\n", encoding="utf-8")
    assert sorted(
        _run(lw._check_perception_stack_crosslink, [p_sam, p_det])["perception_stack_crosslink"]
    ) == [
        "wiki/entities/sam.md",
        "wiki/entities/some-detector.md",
    ]
