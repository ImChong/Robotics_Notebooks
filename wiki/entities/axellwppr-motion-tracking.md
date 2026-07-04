---
type: entity
tags: [repo, humanoid, motion-tracking, teleoperation, mjlab, unitree-g1, tsinghua, shanghai-pil]
status: complete
updated: 2026-07-04
summary: "Axellwppr/motion_tracking：HEFT 官方 mjlab 训练栈，支持 G1/L7 全身跟踪、PMG 配对数据、WPC 负载课程与 sim2real 检查点部署。"
related:
  - ./paper-heft.md
  - ./paper-twist2.md
  - ../methods/sonic-motion-tracking.md
  - ../tasks/teleoperation.md
sources:
  - ../../sources/repos/axellwppr_motion_tracking.md
  - ../../sources/papers/heft_arxiv_2607_02332.md
---

# Axellwppr / motion_tracking

**一句话定义**：[Axellwppr/motion_tracking](https://github.com/Axellwppr/motion_tracking) 是 [HEFT](./paper-heft.md) 的官方 **mjlab** 训练与导出仓库：`main` 实现 G1/L7 **PMG + WPC** 三阶段管线，`sim2real` 提供 **已发布 checkpoint** 与部署运行时。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| HEFT | HEavy-Payload Full-size humanoid Teleoperation | 本仓库 `main` 分支对应论文实现 |
| PMG | Privileged Motion Guidance | `vr_paired/{clean,raw}` 配对 memdataset |
| WPC | Windowed Payload Curriculum | L7 负载课程配置 `cfg/load/l7_wpc.yaml` |
| ONNX | Open Neural Network Exchange | `eval.py --export` 产出部署策略格式之一 |
| GMR | General Motion Retargeting | 配套 [Axellwppr/GMR](https://github.com/Axellwppr/GMR) 生成 50 Hz npz |

## 为什么重要

- **HEFT 可复现入口**：训练 profile（`G1_tracking.yaml` / `L7_tracking.yaml`）、`train.sh` 三阶段与 `eval.py` 导出与论文叙事一致。
- **分支职责清晰**：`sim2real` 单独承载真机部署，避免训练依赖与运行时混杂。
- **PMG 数据契约明确**：clean teacher 与 raw student 分目录存放，便于接入自有 VR 录制流。

## 核心结构

| 组件 | 路径 / 说明 |
|------|-------------|
| 训练入口 | `train.sh` → `run_pipeline "G1_tracking"` 或 `"L7_tracking"` |
| 任务配置 | `cfg/task/profile/G1_tracking.yaml`、`L7_tracking.yaml` |
| WPC | `cfg/load/l7_wpc.yaml`（公开窗标签待发布） |
| 数据集构建 | `scripts/data_process/generate_dataset.py` |
| 评测导出 | `scripts/eval.py`（`-p` 播放，`--export` 导出 ONNX/PT） |
| 部署 | `sim2real` 分支运行时与 checkpoint |

## 与其他页面的关系

- 方法背景：[paper-heft.md](./paper-heft.md)
- 跟踪基线：[paper-twist2.md](./paper-twist2.md)、[sonic-motion-tracking.md](../methods/sonic-motion-tracking.md)
- 任务：[teleoperation.md](../tasks/teleoperation.md)

## 参考来源

- [axellwppr_motion_tracking.md](../../sources/repos/axellwppr_motion_tracking.md)
- [heft_arxiv_2607_02332.md](../../sources/papers/heft_arxiv_2607_02332.md)

## 推荐继续阅读

- [GitHub 仓库](https://github.com/Axellwppr/motion_tracking)
- [HEFT 项目页](https://heft.axell.top/)
- [修改版 GMR 导出器](https://github.com/Axellwppr/GMR)
