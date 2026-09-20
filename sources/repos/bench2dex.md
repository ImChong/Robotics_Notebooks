# Bench2Dex

> 来源归档

- **标题：** Bench2Dex
- **类型：** repo
- **链接：** <https://github.com/Bench2Dex/Bench2Dex>
- **项目页：** <https://bench2dex.github.io/>
- **文档：** <https://bench2dex.github.io/doc/>
- **论文：** <https://arxiv.org/abs/2609.15726>
- **机构：** 上海交通大学（SJTU）· 复旦大学（Fudan）· 香港大学（HKU）等
- **入库日期：** 2026-09-20
- **一句话说明：** 基于 **Isaac Lab v2.3.2 + Isaac Sim 5.1** 的 visuotactile 双臂灵巧操作基准：遥操作采集、HDF5 replay 生成多模态、四策略训练评测与 ModelScope/HF 资产分发。

## 仓库入口（README，2026-09-20）

| 组件 | 说明 |
|------|------|
| 环境 | `conda` + Isaac Sim 5.1.0 + PyTorch 2.7 cu128 + editable Isaac Lab v2.3.2 |
| 资产 | HF/ModelScope：`Assets`、`teleopdata`、`policy_ckpt` |
| 遥操作 | Manus SDK + iPhone ARKit；`python main.py --teleop --collect --enable-generalization --task scenes/<task>.yaml` |
| Replay | `tools/replay/batch_replay.py`：`--enable-rgb --enable-tactile` + 泛化 resample |
| 数据裁剪 | 按 `meta/homing_start_sim_step` 截断回 home 段；四策略管线均遵循 |
| 策略 | ACT / DP / π₀.₅ / GR00T N1.5 — 详见 [Documentation Policy Usage](https://bench2dex.github.io/doc/) |

## 目录布局（预期）

```text
root_path/
├── Bench2Dex/             # 本仓库代码
├── dex2bench_dataset/     # 场景与机器人资产
├── teleopdata/            # 遥操作 HDF5
└── policy_ckpt/           # 预训练 checkpoint
```

## 与仓库内实体的关系

| 关联 | 说明 |
|------|------|
| [paper-bench2dex](../../wiki/entities/paper-bench2dex.md) | 论文实体、扰动轴与评测读法 |
| [bench2dex-github-io](../sites/bench2dex-github-io.md) | 项目页核查 |
| [isaac-lab](../../wiki/entities/isaac-lab.md) | 底层仿真 RL 栈 |
