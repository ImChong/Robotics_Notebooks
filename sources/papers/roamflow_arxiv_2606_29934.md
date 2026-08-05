# RoamFlow: Reinforcement-Aligned One-Step Action MeanFlow Policy for Image-Goal Navigation（arXiv:2606.29934）

> 来源归档（ingest）

- **标题：** RoamFlow: Reinforcement-Aligned One-Step Action MeanFlow Policy for Image-Goal Navigation
- **类型：** paper / navigation / image-goal / generative-policy / meanflow / reinforcement-learning
- **arXiv abs：** <https://arxiv.org/abs/2606.29934>
- **PDF：** <https://arxiv.org/pdf/2606.29934>
- **HTML：** <https://arxiv.org/html/2606.29934>
- **DOI：** <https://doi.org/10.48550/arXiv.2606.29934>
- **机构：** 南洋理工大学（Nanyang Technological University, NTU）
- **作者：** Zixuan Zhang*、Yuqi Chen*、Junjie Gao、Siyuan Song、Yongzhou Pan、Beichen Wang、Mir Feroskhan†（*共一；†通讯）
- **发表 / 上传：** 2026-06-29（arXiv v1）
- **仿真：** Habitat；Gibson 训练 / 验证；MP3D 跨域零微调；专家轨迹 Hybrid A* on NavMesh
- **真机：** Unitree **Go2** + Jetson Orin NX 16GB + RealSense D435i；ROS1 Noetic；控制环 **10 Hz**
- **项目页 / 代码：** 截至入库日 **无**（arXiv Code 区无官方仓；GitHub 检索无 RoamFlow 官方实现）
- **入库日期：** 2026-08-05
- **一句话说明：** 用 **MeanFlow** 预测区间平均速度场做 **一步轨迹生成**，再以 **IL→RL** 两阶段对齐任务目标，并在 Habitat 与 Go2 真机上兼顾 SR/SPL 与低延迟。

## 相关资料（策展）

| 类型 | 链接 | 说明 |
|------|------|------|
| 论文 | [arXiv:2606.29934](https://arxiv.org/abs/2606.29934) | 唯一官方入口 |
| 扩散对照 | [NoMaD](../../wiki/entities/paper-notebook-nomad-goal-masked-diffusion-policies-for-navigat.md) | Table I 多步扩散基线 |
| Critic 轨迹选择对照 | [NavDP](../../wiki/entities/paper-notebook-navdp-learning-sim-to-real-navigation-diffusion.md) | Table I 最强生成基线之一 |
| Image-goal WAM 对照 | [NavWAM](../../wiki/entities/paper-navwam-goal-conditioned-visual-navigation-wam.md) | 世界动作模型路线，非 MeanFlow |
| 仿真栈 | [Habitat-Sim](../../wiki/entities/habitat-sim.md) | 本文仿真评测环境 |

## 摘要级要点

- **问题：** 逐步 RL 近视；扩散 / CFM 生成策略推理慢；蒸馏换速度常损轨迹质量；纯模仿难对齐成功/避碰目标。
- **RoamFlow：** MeanFlow 预测 **interval-level average velocity** → 少步/一步轨迹；EfficientNet-B0 编码 RGB-D 观测与目标图；轨迹评估器重排候选；**IL（专家）→ PPO RL（Habitat）**。
- **仿真（Table I，Gibson 训）：** Gibson SR/SPL/CR/Time = **68.7 / 61.9 / 10.9 / 19.6 ms**；MP3D（未见域）**56.1 / 47.1 / 12.2 / 19.1 ms**；相对 NavDP：SR +9.4、SPL +14.4、时延约 61→20 ms。
- **消融（Table II）：** 无 IL → Gibson SR 13.2；无 RL → 51.3；随机选轨迹 → 60.7；完整 **68.7**。
- **真机（Table III，20 runs / 三场景）：** RoamFlow SR **1.00**、C/run **0.10**、Time **37.2 ms**；优于 FlowNav（0.90 / 0.25 / 89.5）与 NoMaD+Depth。
- **开源状态（截至 2026-08-05）：** **确认未开源**。

## 核心摘录（面向 wiki 编译）

### 1) 系统设定

- Image-goal：无全局定位时，在预建拓扑图上自定位选下一子目标图像；RoamFlow 预测轨迹 waypoint，PD 出 \((v_t,\omega_t)\)。
- 成功：Stop 且位姿距目标观测位姿 \(d_s=1.0\,\mathrm{m}\)、航向 \(\alpha_s=30^\circ\)。
- 数据：Gibson/MP3D + GoStanford/SCAND（GoStanford 深度用 Depth Anything V2）。

### 2) 两阶段训练

| 阶段 | 设置 |
|------|------|
| IL | AdamW \(10^{-5}\)，batch 128，25 epoch，RTX 6000 Ada，约 20 h |
| RL | Habitat PPO；γ=0.99，GAE 0.95；成功 +5 / step −0.01 / 碰撞 −0.1；噪声 σ∈[0.08,0.14]；约 30 h |

### 3) 与邻近路线对照

| 维度 | RoamFlow | NoMaD | NavDP | NavWAM |
|------|----------|-------|-------|--------|
| 生成范式 | **MeanFlow 一步** | 扩散多步 | 扩散 + critic | Cosmos WAM 扩散 |
| 任务对齐 | **IL→RL** | 主模仿 | 特权 critic 筛轨迹 | policy/WM/value 联合 |
| 地图 | 拓扑图局部规划 | 可拓扑记忆 | 仿真特权 ESDF | 非本文设定 |
| 真机 | Go2 Orin NX | 多机器人 | 多本体零样本 | Diablo |
| 开源 | **无** | 已开源 | 已开源 | Coming soon |

## 对 wiki 的映射

- 沉淀实体页：[paper-roamflow.md](../../wiki/entities/paper-roamflow.md)
- 交叉：[NoMaD](../../wiki/entities/paper-notebook-nomad-goal-masked-diffusion-policies-for-navigat.md)、[NavDP](../../wiki/entities/paper-notebook-navdp-learning-sim-to-real-navigation-diffusion.md)、[NavWAM](../../wiki/entities/paper-navwam-goal-conditioned-visual-navigation-wam.md)、[Habitat-Sim](../../wiki/entities/habitat-sim.md)、[VLN 任务](../../wiki/tasks/vision-language-navigation.md)、[VLN 开源复现范式](../../wiki/overview/vln-open-source-repro-paradigms.md)、[Jetson Orin NX](../../wiki/entities/jetson-orin-nx.md)

## 当前提炼状态

- [x] arXiv PDF/HTML 方法与 Table I–III 摘录
- [x] 开源核查：无项目页 / 无代码（步骤 2.5）
- [x] wiki 实体与 NoMaD/NavDP/NavWAM 交叉规划
