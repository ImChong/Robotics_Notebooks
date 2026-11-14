---
type: source
title: "facebookresearch/sparsh — 源码结构与 TacBench 训练评估归档"
url: "https://github.com/facebookresearch/sparsh"
date: 2026-11-14
status: ingested
tags: [repo, tactile-sensing, ssl, vbts, tacbench, meta]
---

# facebookresearch/sparsh — 源码结构与 TacBench 训练评估归档

- **仓库：** <https://github.com/facebookresearch/sparsh>（CC-BY-NC-4.0，已归档，预训练权重托管于 HuggingFace `facebook/sparsh`）
- **开源状态：** 已开源（包含 MAE / DINO / DINOv2 / IJEPA / VJEPA 预训练代码、TacBench 下游任务训练/评测脚手架、实时 DIGIT/GelSight 力场可视化 demo 与 Hydra 配置；下游标注数据与 HuggingFace checkpoints 按文档指引单独下载）。
- **核对范围：** 核对官方仓库 `facebookresearch/sparsh` 源码：
  - `README.md`（安装环境、预训练/微调/测试/Demo CLI 用法、模型与数据集链接）
  - 根入口脚本：`train.py`、`train_task.py`、`test_task.py`、`demo_forcefield.py`
  - Hydra 配置：`config/default.yaml`、`config/experiment/`（`mae_vit.yaml`、`dino_vit.yaml`、`dinov2_vit.yaml`、`ijepa_vit.yaml`、`vjepa_vit.yaml`）、`config/task/`（`t1_force_estimation.yaml`、`t2_slip_detection.yaml`、`t3_pose_estimation.yaml`、`t4_grasp_stability.yaml`、`t6_textile_classification.yaml`、`digit_forcefield.yaml`）
  - 核心模块：`tactile_ssl/downstream_task/sl_module.py`（`SLModule.load_encoder` / `load_task`）、`tactile_ssl/downstream_task/force_sl.py`（`ForceLinearProbe`、`ForceSLModule`）及 `tactile_ssl/test/demo_t1_forcefield.py`（`DemoForceField`）
- **通读：** 已完成

## 关键实现对齐细节

1. **SSL 预训练与下游 Probe 训练解耦入口：**
   - 自监督预训练由 `train.py`（`@hydra.main(..., config_name="default.yaml")`）启动，经 `get_dataloaders(cfg)` 加载多传感器触觉帧对，实例化 `Trainer` 与 `cfg.model`（`mae_vit` / `dino_vit` / `dinov2_vit` / `ijepa_vit` / `vjepa_vit`）完成训练；
   - 下游微调由 `train_task.py` 驱动，支持 `train_data_budget` 控制标签比例，实例化继承自 `SLModule` 的任务模块（如 `ForceSLModule`），通过 `SLModule.load_encoder()` 按 `encoder_type` 自动映射权重前缀（`jepa -> target_encoder`、`dino -> teacher_encoder.backbone`、否则 `encoder`），并默认冻结 `model_encoder` 仅训练 `AttentivePooler` + 任务头（如 `ForceLinearProbe`）。
2. **评估与在线力场可视化链路：**
   - `test_task.py` 加载实验目录 `config.yaml` 与指定 checkpoint，调用 `cfg.test.tester` 执行 `run_model()`、`make_plots()` 与 `get_overall_metrics()`；
   - `demo_forcefield.py` 实例化 `tactile_ssl/test/demo_t1_forcefield.py` 中的 `DemoForceField`，从真机传感器流（`digit_serial` 或 `gelsight_device_id`）采集首帧作为静态背景执行 `compute_diff` 差分拼接为 6 通道张量，分别以 `mode="normal"` 与 `mode="shear"` 前向传播输出实时法向接触掩码与剪切光流矢量图。
3. **命名细节差异提示：** 仓库中织物分类配置文件命名为 `config/task/t6_textile_classification.yaml`，而论文正文中织物分类编号为 `T5`、Bead Maze 机器人操作为 `T6`，阅读代码与论文对照时需留意任务编号映射。
