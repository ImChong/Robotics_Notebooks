# Humanoid-GPT: Scaling Data and Structure for Zero-Shot Motion Tracking

> 来源归档（ingest）

- **标题：** Humanoid-GPT: Scaling Data and Structure for Zero-Shot Motion Tracking
- **类型：** paper
- **venue：** CVPR 2026（项目页标注）
- **原始链接：**
  - arXiv：<https://arxiv.org/abs/2606.03985>
  - 项目页：<https://qizekun.github.io/Humanoid-GPT/>
  - 代码：<https://github.com/GalaxyGeneralRobotics/Humanoid-GPT>
- **机构：** 清华大学；Galbot Inc.；北京航空航天大学；上海交通大学；北京大学；上海期智研究院
- **入库日期：** 2026-06-04
- **一句话说明：** 在 **20 亿帧** G1 重定向语料上，用 **HME 聚类 → 数百 PPO motion expert → 并行 DAgger** 蒸馏为 **GPT 式因果 Transformer** 通用人形 tracker，同时实现 **高动态跟踪** 与 **未见动作零样本泛化**，并给出数据/模型 scaling law；真机 Unitree G1 与 SONIC 对比及 <1.5ms（RTX 4090 + TensorRT）部署叙事。

## 核心论文摘录（MVP）

### 1) 敏捷性 vs 泛化的「伪权衡」与规模假设

- **链接：** <https://arxiv.org/abs/2606.03985> §1
- **摘录要点：** 现有 tracker 多为 **浅层 MLP + 百万～千万级帧**（如 AMASS 约 7.2M），在域内高动态与 **未见风格零样本** 之间长期 trade-off；BeyondMimic / ASAP 偏敏捷、TWIST / UniTracker 偏泛化但难跟极限动态。作者认为瓶颈是 **数据与结构规模不足**，而非任务本质矛盾。
- **对 wiki 的映射：**
  - [Humanoid-GPT](../../wiki/entities/paper-humanoid-gpt.md) — 问题定义与相对 SONIC / TWIST 等定位

### 2) 20 亿帧语料与 HME 多样性度量

- **链接：** <https://arxiv.org/abs/2606.03985> §3
- **摘录要点：**
  - 聚合 **AMASS、LAFAN1、Motion-X++、PHUMA、MotionMillion** 与大规模 **in-house** 采集；过滤坐椅/游泳/爬楼梯等不适配平地 G1 的交互；**General Motion Retargeting** 映射到 **29-DoF Unitree G1**；**时间扭曲** 增广约 **5×** 规模。
  - **Harmonic Motion Embedding (HME)**：多分区 **Periodic Autoencoder** 提取关节谐波幅值/频率 → 序列级 mean/std 聚合 → **K-Means** 得约 **300 簇**（每簇约 1k–2k 条）；用 **gstd** 与 **log-volume** 量化 latent 覆盖，策展集 log-volume 约为 AMASS 的 **4–5×**。
- **对 wiki 的映射：**
  - [Humanoid-GPT](../../wiki/entities/paper-humanoid-gpt.md) — 数据管线、HME 与 diversity-balanced 采样叙事

### 3) 分簇 RL expert + Transformer DAgger 蒸馏

- **链接：** <https://arxiv.org/abs/2606.03985> §4
- **摘录要点：**
  - **Expert**：每簇 **PPO** tracking；观测含特权本体状态 + 参考关节 $q_t^{ref}$；奖励为 **关键点级** 位置/姿态/速度指数项 + 自碰撞/平滑惩罚。
  - **Humanoid-GPT**：将多 expert 行为用 **DAgger** 蒸馏进 **因果 Transformer** $G_\theta$；输入为长度 $H$ 的 $(s_t, q_t^{ref})$ token 序列；**并行多步监督**（式 2）利用因果 mask 一次前向对齐整段 teacher 动作；推理维护 **≤H** 历史队列，取末位输出为当前 PD 目标。
- **对 wiki 的映射：**
  - [Humanoid-GPT](../../wiki/entities/paper-humanoid-gpt.md) — Mermaid 三阶段管线与机制表

### 4) Scaling law、仿真与真机零样本

- **链接：** <https://arxiv.org/abs/2606.03985> §5；项目页对比表
- **摘录要点：**
  - **Table 1**：相对 HumanPlus / OmniH2O / ASAP / GMT / UniTracker / TWIST / Any2Track / **SONIC (100M, MLP)**，Humanoid-GPT 为 **Transformer + 敏捷 + 零样本 + 2.0B 帧**。
  - **Table 2（AMASS-test）**：Humanoid-GPT-L @ 2B tokens SR **92.58%**；同数据下 MLP/TCN 在 2B 上增益饱和且小数据大模型易过拟合。
  - **真机**：训练外舞蹈序列零样本跟踪；在线 MoCap→G1 retarget **全身遥操作**；相对 SONIC 在 daily/dance/high-dynamic/balance 四类对比中站点主张更平滑/更稳。
  - **部署**：ONNX + TensorRT，RTX 4090 端到端 **<1.5ms**（约 **5×** 快于 TWIST 叙述）。
- **对 wiki 的映射：**
  - [Humanoid-GPT](../../wiki/entities/paper-humanoid-gpt.md) — 实验指标、与 SONIC 对照、部署与局限

## 对 wiki 的映射（汇总）

- [paper-humanoid-gpt.md](../../wiki/entities/paper-humanoid-gpt.md) — 主沉淀页
- 交叉更新：[sonic-motion-tracking.md](../../wiki/methods/sonic-motion-tracking.md)、[humanoid-motion-tracking-method-selection.md](../../wiki/queries/humanoid-motion-tracking-method-selection.md)

## 引用（项目页 BibTeX）

```bibtex
@article{humanoidgpt26,
  title={Humanoid-GPT: Scaling Data and Structure for Zero-Shot Motion Tracking},
  author={Qi, Zekun and Chen, Xuchuan and Liu, Dairu and Lin, Chenghuai and Lian, Yunrui and Liang, Sikai and Zhang, Zhikai and Guan, Yu and Wang, Jilong and Zhang, Wenyao and Yu, Xinqiang and Wang, He and Yi, Li},
  journal={arXiv preprint arXiv:2606.03985},
  year={2026}
}
```
