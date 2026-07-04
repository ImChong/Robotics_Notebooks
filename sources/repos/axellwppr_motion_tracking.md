# Axellwppr/motion_tracking（HEFT 人形全身运动跟踪训练栈）

- **标题**: motion_tracking — HEFT 官方训练 / 评测 / 导出实现
- **类型**: repo / humanoid / motion-tracking / teleoperation / mjlab
- **作者**: Axellwppr（HEFT 作者团队维护）
- **机构**: 清华大学、RobotEra、上海期智研究院（见 [HEFT 论文](../papers/heft_arxiv_2607_02332.md)）
- **链接**: <https://github.com/Axellwppr/motion_tracking>
- **项目页**: <https://heft.axell.top/>
- **许可证**: MIT
- **收录日期**: 2026-07-04

## 一句话摘要

基于 **mjlab** 的人形 **全身运动跟踪** 训练框架：`main` 分支实现 [HEFT](https://heft.axell.top/) 的 **PMG + WPC** 与 G1/L7 三阶段 `train→adapt→finetune` 管线；`sim2real` 分支提供 **已发布 checkpoint** 与部署运行时；`compliance` 分支为 G1 柔顺跟踪扩展。

## 为何值得保留

- **重载 VR 遥操作可复现栈**：公开 G1 烟测 memdataset、训练 profile、导出 ONNX/PT 与 sim2real 部署，降低 HEFT 复现门槛。
- **PMG 数据形态明确**：`vr_paired/{clean,raw}` 配对 memdataset 约定，teacher 侧 clean、student 侧 raw，与论文 PMG 叙事一致。
- **多机器人 profile**：`G1_tracking.yaml` 与 `L7_tracking.yaml` 并列，体现 **全尺寸 L7** 与 G1 共用框架。
- **生态衔接**：配套 [Axellwppr/GMR](https://github.com/Axellwppr/GMR) 生成 GMR 风格 50 Hz `npz`；与 TWIST2、SONIC 等 tracking 基线在同一评测叙事中对齐。

## 分支与发布状态

| 分支 | 用途 |
|------|------|
| `main` | HEFT 官方训练实现（PMG、WPC、G1/L7 profile） |
| `sim2real` | 部署运行时、sim2sim/sim2real 配置、**已发布 checkpoint** |
| `compliance` | G1 tracking + 柔顺框架（Gentle humanoid 启发）；[演示站](https://motion-tracking.axell.top/) |

**已发布：** 高效跟踪框架、PMG/WPC、`sim2real` checkpoint 与运行时。  
**待发布：** 完整训练集、WPC 窗负载标签、VR 录制/重建/配对数据生成工作流。

## 技术要点

### 环境与依赖

- Python 3.10；依赖管理 **uv**（`uv sync`）。
- 仿真 / 训练基于 **mjlab 1.3.0**。

### 数据集布局

- 默认根 `dataset/` 或环境变量 `MEMPATH`。
- G1/L7 均支持 `lafan`、`100style`、`seed/all`、`vr_paired/{clean,raw}` 等组。
- PMG：clean 为 teacher 目标，raw 为 deployable student 输入。
- 烟测数据：[Google Drive 样例](https://drive.google.com/drive/folders/1-FBUxllaYwqGIUSCaWg_4inD-u5Tdvi9?usp=sharing)（非完整 HEFT 训练集）。

### 训练入口

- `train.sh` 配置 `PROJECT`、`NPROC` 与 `run_pipeline`。
- G1：`run_pipeline "G1_tracking" "g1_track" "release"`
- L7：`run_pipeline "L7_tracking" "l7_track" "release"`
- 三阶段：`+exp=train` → `+exp=adapt` → `+exp=finetune`；8×8192 env 约 6 h（8×PRO6000）。

### 评测与导出

- `scripts/eval.py --run_path ... -p` 播放；`--export` 导出 `policy.onnx` / `policy.pt` / `policy.json`。
- L7 评测：`--profile L7_tracking`。

## 对 Wiki 的映射

- **wiki/entities/paper-heft.md**：HEFT 论文实体页（方法、真机负载、流程图）。
- **wiki/tasks/teleoperation.md**：VR 重载全身遥操作工程参考。
- **wiki/entities/paper-twist2.md**、**wiki/methods/sonic-motion-tracking.md**：跟踪基线对照。
