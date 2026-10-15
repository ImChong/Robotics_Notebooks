# Modeling and Understanding Neural Weights of Humanoid Policies with Diffusion（arXiv:2610.20606）

> 来源归档（ingest）

- **标题：** Modeling and Understanding Neural Weights of Humanoid Policies with Diffusion
- **类型：** paper / humanoid / weight-space generative model / diffusion / Unitree G1
- **arXiv：** <https://arxiv.org/abs/2610.20606>（v1，提交 2026-10-14，cs.RO；HTML：<https://arxiv.org/html/2610.20606v1>）
- **项目页：** <https://humanoid-weight-diffusion.github.io/>
- **代码：** 未发布（项目页标注 "Code · coming soon"，未给仓库链接；页面仅引用一个第三方项目页模板的 GitHub 链接，非作者代码）
- **作者：** Huixuan Cheng, Bike Zhang, Shankar Sastry, Koushil Sreenath（通讯作者 Sreenath）
- **机构：** University of California, Berkeley（据项目页作者列表）
- **入库日期：** 2026-10-15
- **一句话说明：** 把强化学习得到的人形策略 actor 网络权重压缩到线性自编码器潜空间，再用 Diffusion Transformer 生成可部署的权重；在 G1 上展示行走、跳舞与旋踢，并分析权重空间的运动学结构。

## 开源状态（步骤 2.5）

- **项目页核查（2026-10-15）：** "Code · coming soon"；无 GitHub、Hugging Face、数据或权重链接；"Watch the demo" 无可见链接目标。
- **arXiv 页：** 未给代码链接。
- **结论：** **待发布**。

## 摘录 1：问题与主张（摘要）

- **问题：** 高性能人形控制网络能否被生成式地建模其权重。
- **方法：** 线性自编码器压缩 actor 参数到紧凑潜空间；Diffusion Transformer 建模潜变量分布；生成的网络可直接部署。
- **结果（摘要）：** 在 G1 真机上 zero-shot 部署生成网络，完成行走、跳舞、旋踢。
- **分析：** 第一层权重反映机器人运动学结构；该结构在 RL 与模仿学习下都出现，提示它来自人形本体而非特定训练方法。

## 摘录 2：方法与评测（arXiv HTML v1，摘要性抽取）

- **数据：** mjlab 中 PPO + 域随机化训练 G1 策略，750 次带种子训练，每次保留 8 个收敛后 checkpoint，每个动作保留 6,000 个 actor；actor 为三层 MLP（[128,128]，ELU，约 3–4 万参数）；只建模 actor 权重。
- **线性自编码器：** 编码/解码各为无偏置、无激活的单层线性层，每个动作单独训练；第一层 (w1,b1) 与后两层 (w2,b2,w3,b3) 分别映射到 1024 维潜变量，拼接为 2048 维。
- **DiT：** 潜空间 x-prediction；embedding 1024，depth 8，8 heads，patch 16，序列长 128；无条件生成（仅 timestep 经 adaLN-Zero）；采样 1000 步。
- **IL 对照：** 每个动作由 50 个 RL 教师蒸馏 50 个 IL 网络（MSE 回归）。
- **任务：** 行走（速度指令）、跳舞与旋踢（动作跟踪）。
- **仿真结果（DiT 均值 vs 数据集）：** 行走成功率 0.999 / 数据集 0.999，跟踪误差 0.162；跳舞成功率 0.998，误差 0.037；旋踢成功率 0.997，误差 0.046。
- **记忆性检验：** 扩散生成优于同距离噪声扰动训练网络（行走 0.993 vs 0.914；跳舞 0.941 vs 0.865；旋踢 0.943 vs 0.783）。
- **IL vs RL：** 成功率 0.9979 vs 0.9994（行走）、0.9948 vs 0.9973（跳舞）、0.9919 vs 0.9972（旋踢）。
- **真机：** zero-shot 部署；未找到定量真机指标。
- **作者陈述的局限：** 权重空间分析只覆盖第一层；只研究有限动作与单一本体，泛化未验证；可解释结构尚未带来性能提升；未来计划做深层分析与条件生成（行为、轨迹或文本）。

## 建议 wiki 动作

- 升格 [`wiki/entities/paper-humanoid-policy-weight-diffusion.md`](../../wiki/entities/paper-humanoid-policy-weight-diffusion.md)：流程总览（RL 权重→自编码器→DiT→解码→真机），并强调「每个动作单独建模，不能推断任意新技能生成」。
- 开源状态写为「待发布」。
