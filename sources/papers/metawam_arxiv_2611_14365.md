# From Video Frames to Robot Meta-Frames: A Unified Latent Interface for World Action Models / 从视频帧到机器人元帧：世界–动作模型的统一潜空间接口

> 原始论文来源归档；这里只归纳论文的公开方法、数据与评测边界，不替代全文。

- **类型：** paper
- **原文：** https://arxiv.org/abs/2611.14365
- **全文：** https://arxiv.org/html/2611.14365v1
- **版本：** arXiv:2611.14365v1（2026-11）
- **核对范围：** HTML 全文：Introduction、Related Work、Motivating Analysis、Method 4.1–4.5、Experiment 5.1–5.3、Conclusion；Appendix A–C（跨模型诊断、训练设置、方法讨论），以及核心对照表。
- **通读：** 已完成
- **收录日期：** 2026-11-12
- **官方代码：** https://github.com/showlab/MetaWAM

## 论文概要

MetaWAM 将机器人动作、状态和价值信号塞入与视频扩散兼容的频域 meta-frame，形成统一世界–动作潜空间接口。

## 机制要点

- 采用正交 2D-DCT 频域 meta-frame 分槽编码 robot proprio、action、future proprio 和 value。
- 官方开源 README 给出 RoboCasa 24 任务、每任务 50 次、3 个随机种子的评测流程。
- 开放源码、RoboCasa 权重及训练脚本；实际评测尚未在本环境复现。

## 评测边界

论文 RoboCasa 24 任务平均成功率 68.9%（对照 Cosmos Policy 67.1%），推理延迟约 0.479 s → 0.291 s（约 -39%）；Franka 三项真机任务平均成功率 85.0%，OOD 设置 68.3%。这些数值是论文自报；时延不等于机器人低层控制周期。

## 通读原文：方法、实验与附录复核

### 元帧结构与反变换

不是简单把全部信号合为一张图，而是 **1 帧输入 meta-frame + 6 帧多相机观测 + 1 帧输出 meta-frame，共 8 帧 latent**；相比逐信号分帧基线的 11 帧缩短序列。机器人任务、当前 proprio、动作块、未来 proprio、value 占据互斥通道的 **2D-DCT 第 0 行系数槽**；IDCT 提升为视频 latent，推理后通过 DCT 解析还原，**无学习式动作解码头**。slot-aware mask 依 policy / world / value 三种任务切换条件与监督槽。

### 实验与效率

表 II：RoboCasa 24 任务平均 **68.9% vs Cosmos Policy 67.1%**，各任务每场景 50 次，3 seeds，共 **3600** trials；表 V：序列 11→8 latent frames，单次查询 **0.479s→0.291s（约 39%）**。表 III：LIBERO **98.1%**、LIBERO-Plus **83.4%**，但 robot initial-state 和 background 两类仍弱于 Cosmos Policy。表 IV：Franka 三任务（每任务 20 次）**85.0% ID / 68.3% OOD**。

### 附录与工程约束

RoboCasa 基座 Cosmos-Predict2-2B，4×H200、64h、global batch 1024、12000 steps；动作 chunk 32、开环执行 16。源仓 scripts/train_robocasa.sh、scripts/test_robocasa.sh 对齐实际训练评测入口；开源权重不代表本次已在本地复现。附录 C 说明 fixed frequency band 与 loss weight 难以覆盖所有任务比例，语言接地能力仍受视频骨干限制。

## 仍待独立验证

- 论文成功率与延迟属于作者报告结果，尚未在本地复现实验；需检查代码和数据版本是否可获取。

## 方法–源码交叉核验（2026-11-12）

- **论文 §4.2（§5–6 页）：** 将 task、current proprio、action chunk、future proprio、value 写入互斥通道频域系数的 `k_H=0` 行，**输入使用 IDCT（频域→空间）**，去噪后使用 **DCT（空间→频域）+ Unpack**；不得反写成 “逆 DCT 解码动作”。
- **论文 §4.3–4.4（§6 页）：** 两个元帧加六个多视角 latent 时间槽共 8 帧，按 policy/world/value 在槽位级切换条件与预测目标；输入元帧复用预留空白槽。
- **论文 §4.5、表 2–5：** diffusion loss 联合动作/未来本体/value 等辅助损失；RoboCasa 68.9%（Cosmos Policy 67.1%），查询时延 0.291 s（对照 0.479 s）；LIBERO Original 98.1% **低于** Cosmos Policy 98.5%，LIBERO-Plus 83.4% 高于 82.2%；真机 85.0% ID / 68.3% OOD。
- **附录 A–C：** Wan2.1/Wan2.2 仅做 frozen-denoiser 表征兼容性分析，**不能表述为在 Wan 骨干上完整训练并复现 RoboCasa 指标**。固定频带、损失权重和语言接地是明确局限。
- **源码对应：** `cosmos_policy/models/policy_text2world_model.py` 中 `_encode_frequency_meta_frame_v2` 与 `_dct2_per_channel_ortho`；`cosmos_policy/models/policy_video2world_model.py` 中条件槽 mask；`cosmos_policy/experiments/robot/cosmos_utils.py` 中动作读出；额外的 `slot_mask_embedder` 位于 DiT 实现。详见 [官方源码归档](../repos/metawam.md)。

这轮为方法/实现核验；论文数值均以作者实验为准，未运行 GPU 训练或复现实验。

## 对 wiki 的映射

- [MetaWAM](../../wiki/entities/paper-metawam.md)
