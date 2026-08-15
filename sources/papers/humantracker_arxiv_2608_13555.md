# HumanTracker: Towards Comprehensive and Human-Aligned Motion Tracking Benchmark

> 来源归档（ingest）

- **标题：** HumanTracker: Towards Comprehensive and Human-Aligned Motion Tracking Benchmark
- **类型：** paper
- **venue：** ECCV 2026（官方仓库 README 标注）
- **原始链接：**
  - arXiv：<https://arxiv.org/abs/2608.13555>
  - 项目页：<https://dairuliu.github.io/humantracker/>
  - 代码：<https://github.com/GalaxyGeneralRobotics/HumanTracker>
- **机构：** 南开大学；清华大学；银河通用（Galbot）；上海交通大学；北京大学；上海期智研究院
- **入库日期：** 2026-08-15
- **最近复核：** 2026-08-15
- **一句话说明：** 用 **153 h / 25K** 四族光学动捕基准 + 偏好对齐指标 **HumanScore**，纠正「MPJPE 好看但视频难看」的人形 motion tracking 评测错位；零样本对照 GMT / TWIST2 / SONIC / Humanoid-GPT。

## 核心论文摘录（MVP）

### 1) 运动学误差与人眼判断错位

- **链接：** <https://arxiv.org/abs/2608.13555> §1
- **摘录要点：** 主流评测把 tracking 当成逐帧姿态匹配，再对关节与时间取平均（MPJPE / keypoint error）。这会漏掉支撑不稳、脚滑、接触时序错位、失败后恢复等**视频里最刺眼**的物理伪影。两段 rollout 可以 MPJPE 接近，观察者仍稳定偏好接触干净、支撑稳定的那条。常用测试集仍是 **AMASS 约 140 条**，覆盖不足且常压成单一总分。
- **对 wiki 的映射：**
  - [HumanTracker](../../wiki/entities/paper-humantracker.md) — 问题定义与评测错位

### 2) 四族 153 小时光学基准

- **链接：** <https://arxiv.org/abs/2608.13555> §3.1、Table 1–2、Appendix D
- **摘录要点：**
  - **24** 名职业表演者（舞蹈/健身/网球教练与专职动捕演员）棚拍多相机光学轨迹；经 [GMR](../../wiki/methods/motion-retargeting-gmr.md) 重定向到 **29-DoF** 人形，并剔除漂浮、穿地、接触不连续片段。
  - 每条 clip 含族标签、自然语言描述、拟合 **SMPL** 与机器人 `qpos` 参考。
  - 四族（小时 / clip）：Daily **89.29 / 9,739**；Highly Dynamic **11.01 / 2,676**；Interaction **47.78 / 10,940**；Ground **4.59 / 1,640**。合计约 **153 h / 25K**。
  - 按源动作 **9:1** 划分：训练 **22,495** 条 / 测试 **2,500** 条；同 `motion_id` 不跨分区。相对 AMASS（>40 h，无类别/无文本）、HumanML3D（28.6 h，有文本无类别）、PHUMA（73 h，无类别/无文本），HumanTracker 同时提供 **类别 + 文本**。
- **对 wiki 的映射：**
  - [HumanTracker](../../wiki/entities/paper-humantracker.md) — 数据集规模与诊断族
  - [人形参考运动数据集选型](../../wiki/comparisons/humanoid-reference-motion-datasets.md) — 与 AMASS / PHUMA 对照

### 3) 标准化评测协议与零样本表

- **链接：** <https://arxiv.org/abs/2608.13555> §3.2、Table 3
- **摘录要点：** 统一 **29-DoF `qpos` + 共用 MuJoCo 入口**；各 tracker 保留原生观测/动作解码。记录 50 Hz 状态史（`qpos`/`qvel`、动作、电机目标、足接触与力、14 关键点）。终止准则对齐 [SONIC](../../wiki/methods/sonic-motion-tracking.md)：骨盆/踝/腕垂向误差 > **0.25 m**、骨盆旋转 > **1 rad** 或非有限值即失败。指标：**Succ**、**MPJPE**（29 主动关节 rad）、**HumanScore**（0–100）。四方法均**不在 HumanTracker 上训练**。
  - Daily：Humanoid-GPT Succ **94.4** / MPJPE **0.046** / HS **54.7**；SONIC 93.8 / 0.102 / 49.5；TWIST2 60.1 / 0.105 / 10.1；GMT 17.0 / 0.250 / 2.4。
  - Highly Dynamic：Humanoid-GPT **86.9 / 0.047 / 49.2** 领先。
  - Interaction：SONIC Succ **97.6** 最高；Humanoid-GPT HS **56.8** 最高。
  - Ground：GMT/TWIST2 Succ **0.0**；Humanoid-GPT Succ **32.9**；SONIC HS **26.5** 最高（感知更稳）。
- **对 wiki 的映射：**
  - [HumanTracker](../../wiki/entities/paper-humantracker.md) — 协议与 Table 3
  - [Humanoid-GPT](../../wiki/entities/paper-humanoid-gpt.md)、[SONIC](../../wiki/methods/sonic-motion-tracking.md)、[GMT](../../wiki/entities/paper-gmt.md)、[TWIST2](../../wiki/entities/paper-twist2.md)

### 4) HumanScore：轨迹级偏好奖励模型

- **链接：** <https://arxiv.org/abs/2608.13555> §3.3–3.5、§4.3–4.4、Appendix A–C
- **摘录要点：**
  - 仅用**训练 split** 上 GMT / Humanoid-GPT / SONIC / TWIST2 的对齐 rollout；切 **250 帧 / 5 s** 窗。均匀抽样后六种 tracker 配对各占一份。6 名人形方向博士标注 **6,000** 原始对（严格偏好 / Similar / Cannot compare）；左右镜像得 **12,000** 条。Cannot compare 不进训练；按 `motion_id` 80/20 划分。
  - 每帧 **539 维** token（当前参考 70 + 仿真 rollout 469：状态/动作、测量接触、根运动、14 关键点）。**不用未来参考残差**。4 层 Transformer + mask mean pooling → 标量无界奖励；严格对用 Bradley–Terry，Similar 用对称损失。
  - 推理：窗奖励经 sigmoid 后按真实帧数加权，映射到 **0–100**。
  - 族均衡 Align Rate：HumanScore **90.83%**（95% CI 87.36–93.83），高于 KPT MAE **84.05**、MPJVE **84.04**、MPJPE **80.49**、足接触 **78.82**。去掉测量接触特征在 Ground 上掉得最狠；加长到 5 s 上下文才能看见滑步/抖动/漂移/恢复。
- **对 wiki 的映射：**
  - [HumanTracker](../../wiki/entities/paper-humantracker.md) — HumanScore 机制与对齐数字

## 开源核查（步骤 2.5，2026-08-15）

| 资源 | 项目页按钮 | 实际入口 | 结论 |
|------|------------|----------|------|
| 论文 | Coming Soon | arXiv:2608.13555 已上线 | 预印本可读 |
| 代码 | Coming Soon | [GalaxyGeneralRobotics/HumanTracker](https://github.com/GalaxyGeneralRobotics/HumanTracker)（Apache-2.0） | **评测框架 + HumanScore 训练/权重已开** |
| 数据集 | Coming Soon | README 要求本地 `--mocap_path` / `HUMANTRACKER_DATASET`；仓内无 25K NPZ | **153 h 基准未发布** |

仓内可辨识入口：`python -m humantracker.eval.eval_parallel_tracker`、`src/humantracker/eval/eval.sh`、`python -m humantracker.reward_model.train.trainer`、`storage/checkpoints/reward_model/best.pt`（约 40 MB）。四个 tracker 经 `setup_thirdparty.sh` 钉提交拉取，不在本仓。

## 对 wiki 的映射（汇总）

- [paper-humantracker.md](../../wiki/entities/paper-humantracker.md) — 主沉淀页
- 交叉更新：[paper-humanoid-gpt.md](../../wiki/entities/paper-humanoid-gpt.md)、[sonic-motion-tracking.md](../../wiki/methods/sonic-motion-tracking.md)、[paper-gmt.md](../../wiki/entities/paper-gmt.md)、[paper-twist2.md](../../wiki/entities/paper-twist2.md)、[humanoid-motion-tracking-method-selection.md](../../wiki/queries/humanoid-motion-tracking-method-selection.md)、[humanoid-reference-motion-datasets.md](../../wiki/comparisons/humanoid-reference-motion-datasets.md)

## 引用（项目页 / 仓库 BibTeX）

```bibtex
@misc{liu2026humantrackercomprehensivehumanalignedmotion,
  title         = {HumanTracker: Towards Comprehensive and Human-Aligned
                   Motion Tracking Benchmark},
  author        = {Dairu Liu and Zekun Qi and Jiayu Zeng and Ruixi Yu and
                   Yu Guan and Yintianrun Zhang and Xuchuan Chen and
                   Sikai Liang and Zekai Li and Chenghuai Lin and
                   Xinqiang Yu and Wenyao Zhang and He Wang and Li Yi},
  year          = {2026},
  eprint        = {2608.13555},
  archivePrefix = {arXiv},
  primaryClass  = {cs.RO},
  url           = {https://arxiv.org/abs/2608.13555}
}
```
