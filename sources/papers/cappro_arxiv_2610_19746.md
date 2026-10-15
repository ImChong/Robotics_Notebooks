# One-shot Human Video-Guided Adaptation for Vision-Language-Action Models（CapPro，arXiv:2610.19746）

> 来源归档（ingest）

- **标题：** One-shot Human Video-Guided Adaptation for Vision-Language-Action Models
- **缩写 / 框架：** CapPro
- **类型：** paper / VLA / video-conditioned adaptation / one-shot / manipulation
- **arXiv：** <https://arxiv.org/abs/2610.19746>（v1，提交 2026-10-14，cs.RO；HTML：<https://arxiv.org/html/2610.19746v1>）
- **项目页：** <https://chenqi20020226.github.io/cappro-project/>
- **代码：** 未发布（项目页 Code 入口标注 "Coming soon"）
- **作者：** Qi Chen, Yuwei Wu, Quan Chen, Kunpeng Gao, Jin Chen, Che Sun, Zhi Gao, Yunde Jia
- **机构：** 未核实（arXiv 摘要页与项目页均未列机构）
- **入库日期：** 2026-10-15
- **一句话说明：** 用一段人类示范视频引导冻结或微调后的 VLA 适配新任务，无需目标任务机器人示范或微调；核心是阶段描述监督的视频压缩表示与视频条件适配模块。

## 开源状态（步骤 2.5）

- **项目页核查（2026-10-15）：** Paper、Code、Demo 三个入口均标注 "Coming soon"；页面正文写明 "We are preparing the paper, source code, and demonstration videos."；未列 GitHub、Hugging Face 或数据/权重链接。
- **arXiv 页：** 未给代码仓库链接。
- **结论：** **待发布**（宣称将发布，入库日未见可获取的代码或权重）。

## 摘录 1：问题与主张（摘要）

- **问题：** VLA 适配新任务通常需要目标任务的机器人示范与微调；论文探索仅凭一段人类视频完成 one-shot 适配。
- **方法：** 两部分——(i) 阶段感知模块：学习视觉 token 选择，并用阶段描述（phase caption）监督，把视频压缩为紧凑表示；(ii) 视频条件适配模块：把该表示接入预训练策略的动作条件上下文。
- **立场：** 人类视频提供的是任务层面的执行指导，而非需要逐帧模仿的轨迹。
- **结果（摘要）：** 未见目标任务平均性能提升，源任务表现相当。

## 摘录 2：方法（arXiv HTML v1，摘要性抽取）

- **基座：** 预训练 π0.5，LoRA 适配 2B VLM 主干与 300M 动作专家。
- **视频编码：** 均匀采样 8 帧，经 VLA 共享视觉编码器编码；可学习 GridS 采样器每帧保留 32 个空间 token；每帧归一化进度正弦编码后加入 token，形成压缩表示 Z^H。
- **查询分支：** 4 个可学习查询注意机器人池化视觉特征与 Z^H；训练期加入教师强制的阶段 caption（结构化 mask 防止查询看到 caption token），输出经可学习标量门控。
- **注入：** 动作前缀拼接机器人特征、Z^H、门控查询输出与语言/状态特征，再由 flow-matching 动作生成器输出动作。
- **训练损失：** 动作损失 + 0.1 × caption 损失；推理时不需要阶段 caption 与进度标签。

## 摘录 3：实验（arXiv HTML v1，摘要性抽取；数值待与原表逐格复核）

- **仿真：** RoboTwin，39 个任务（源 13；Target-A1 6；Target-A2 10；Target-B 10），每设定 100 次 rollout（clean / randomized）。
- **真机：** 7 个任务（源 3；Target-A 2；Target-B 2），每任务 5 次试验；真机机器人型号未核实。
- **RoboTwin 成功率（%，摘录）：**

| 方法 | Source | A1 | A2 | B | Overall |
|------|--------|----|----|---|---------|
| π0.5 | 64.4 | 34.5 | 2.5 | 4.0 | 28.4 |
| Direct Video Prefix | 43.5 | 25.2 | 1.5 | 1.4 | 19.1 |
| CapPro | 68.9 | 44.8 | 29.0 | 20.7 | 42.6 |

- **真机：** 摘录中 π0.5 与 CapPro 的逐方法列未能可靠区分，真机逐方法数值未核实。
- **局限（作者陈述）：** 大幅物体、交互或组合性变化的任务仍弱（A2 29.0%、B 20.7%）；计划改进视频表示与适配方法。

## 建议 wiki 动作

- 升格 [`wiki/entities/paper-cappro.md`](../../wiki/entities/paper-cappro.md)：一句话定义、流程总览（视频→压缩表示→适配前缀→动作）、与 VLA 适配 / in-context 类工作对比。
- 开源状态写为「待发布」；代码出现后再升级。
