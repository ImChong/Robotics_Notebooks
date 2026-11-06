# HALO：Perceptive Humanoid Motion from Simulation（arXiv:2611.06908）

> 来源归档（paper）

- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2611.06908>（核对时仅有 **v1**，提交于 2026-11-04）
- **PDF：** <https://arxiv.org/pdf/2611.06908>（20 页，已通读正文与附录 A–G）
- **标题：** Perceptive Humanoid Motion from Simulation（系统名 HALO：Humanoid Autonomy, Learned in simulation, running Onboard）
- **作者：** Siyuan Hu, Abulikemu Abuduweili, Yue Meng, Shipeng Liu, Quentin Leboutet, Chen Chen, Matthias Müller, Vladlen Koltun（Apple）
- **分类：** cs.RO；cs.AI；cs.CV；cs.LG；arXiv 备注 "Code and videos at" 项目页
- **项目页：** [sources/sites/halo-perceptive-humanoid-motion-sim.md](../sites/halo-perceptive-humanoid-motion-sim.md)
- **代码：** <https://github.com/apple-aiml-research/ml-halo> — [sources/repos/halo-perceptive-humanoid-motion-sim.md](../repos/halo-perceptive-humanoid-motion-sim.md)
- **入库日期：** 2026-11-06
- **线索来源：** [2026-11-06-seven-robotics-intake-draft.md](2026-11-06-seven-robotics-intake-draft.md)

## 与库内同名节点的区分

名称相近但不是同一篇：`2603.15084`（HALO，重载人形 sim-to-real）、`2606.18772`（HALOMI）、`2607.04265`（HALO-WA）。本篇 arXiv ID 为 `2611.06908`。

## 核心摘录（原文数字，括号为出处）

- 摘要：在一千个程序生成的住宅里合成视觉-语言-运动数据；真机在 6 个未建图环境、360 次试验取得 91% 成功率（abstract）。
- 三层栈：编排器约 0.5 Hz、运动生成器约 3 Hz、运动跟踪器 50 Hz（Fig. 2）；运动计划为 5 s、30 Hz、每帧 36 维（29 关节 + 3 平移 + 4 四元数）（§III）。
- 语料：6 个技能数据集共 162,698 个 episode、296.3 h（附录 A Table IV）；正文取整为 300 h；合成耗时 48 h / 250 张 H100，生成器训练 64 张 B200，跟踪器 4 张 H100（§IV-A）。
- 生成器：冻结 Qwen3.5-2B（64 图像 token + ≤96 文本 token）+ 8 块 DiT（宽 512），可训练 49.3M，flow matching，15 步 Euler，前 P=15 帧为干净前缀（§III-B、附录 B Table V）。
- 跟踪器：三阶段 = 特权 PPO 专家 → 多教师蒸馏（BC + DAgger，20% 教师）→ 多 critic PPO（附录 C Table VI–VIII）。
- 单任务成功率（Table II）：仿真均值 91、真机均值 91；多任务均值 仿真 69、真机 70。
- 消融（Table III，SIT，100 留出仿真环境）：训练场景 1/10/100/1,000 = 43/64/81/84；RTC 无/推理时/训练时 = 24/66/84；回归解码器 19；无深度 74。
- 机载延迟（Table IX）：生成器约 152 ms；编排器一次转移决策约 0.72 s（单独）/ 约 1.46 s（与生成器共用 GPU）。
- 硬件：Unitree G1 + OAK-1W（120° RGB 1280×800）+ RealSense D435i（55° 深度 424×240，经 LiteAnyStereo）+ Mac mini M4 Pro 64 GB + 机内 Jetson Orin NX。
- 局限（§V）：新任务仍需人工写提示与关键帧；无触觉；生成器-跟踪器接口窄；编排层较简单。

## 已发现的出入 / 笔误

- 草稿登记（v1 摘要级）未写：真机 360 次 = 6 技能 × 6 环境 × 10 次；91% 为 6 项单任务均值；多任务真机均值仅 70%；仓库已公开代码但不含权重与数据。
- 正文「300 小时」与 Table IV 的 296.3 h 为取整关系。
- 附录 E 写 OAK-1W 为 1280×80，应为 1280×800（正文 §IV-B 为 1280×800）。

## 对 wiki 的映射

- [paper-halo-perceptive-humanoid-motion-sim](../../wiki/entities/paper-halo-perceptive-humanoid-motion-sim.md)
