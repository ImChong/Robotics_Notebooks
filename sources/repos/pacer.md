# ChangChrisLiu/PACER

> 来源归档

- **标题：** PACER（Process-Aware Correction and Evidence Reweighting）官方仓库
- **类型：** repo
- **代码：** <https://github.com/ChangChrisLiu/PACER>
- **论文：** [arXiv:2610.19248](https://arxiv.org/abs/2610.19248) — 归档见 [`sources/papers/pacer_arxiv_2610_19248.md`](../papers/pacer_arxiv_2610_19248.md)
- **项目页：** <https://pacer-corl.github.io/> — 归档见 [`sources/sites/pacer-project.md`](../sites/pacer-project.md)
- **入库日期：** 2026-10-17（核查日同日）
- **许可：** MIT（Copyright 2026 Chris Liu；仅覆盖代码、schema、示例与文档）
- **一句话说明：** 研究框架 + 运行时适配参考：通用 `PACER_Framework/`（不引硬件）与 `pacer/` + `scripts/` 适配层；仅合成示例，无真实数据 / 检查点 / 机器人驱动。

## 目录（README 所列根目录）

`.github/workflows/`、`PACER_Framework/`、`configs/`、`docs/`、`examples/`、`pacer/`、`scripts/`、`tests/`、`CHANGELOG.md`、`LICENSE`、`README.md`、`pyproject.toml`。

## 复现入口（README 摘要，未本地运行）

```bash
# PACER_Framework/ 下
python -m pip install -e '.[dev]'
python -m pacer_framework.check_setup
python examples/end_to_end_demo.py
python -m pacer_framework.demo --out-dir .local/demo-default
# 仓库根目录，无硬件
python scripts/hardware/run_policy_inference.py --backend mock --mock-observation --prompt "move to the target" --output /tmp/pacer_mock_infer.json
python scripts/robot_runtime/run_correction_collection.py --dry-run
python scripts/robot_runtime/run_rollout_eval.py --dry-run
```

README 九步工作流：基座 VLA 检查点 → 记录 rollout → 采集纠正与演示 → 校验来源与安全 → 计算过程证据 → 导出加权行与基线 → 外部训练 → 固定验证行评测 → 选审计通过的候选。安全提示：先 `--dry-run`；实机运动需显式确认、有效安全栈与物理急停。

## 与本仓库知识的关系

[PACER 实体页](../../wiki/entities/paper-pacer.md)
