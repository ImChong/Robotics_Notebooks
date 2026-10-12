# OmniDex 论文来源归档

## 元数据

- **题目：** OmniDex: Scaling Dexterous Hand Grasping to Diverse Cluttered Scenes
- **作者：** Naiyu Fang, Zhongjin Luo, Yuxin Mo, Siyuan Huang, Jianbo Liu, Yufei Liu, Zheyuan Zhou, Chenkai Jin, Xiaogang Wang, Hongsheng Li
- **单位（据 HTML 版）：** ACE Robotics；CUHK MMLab；CPII under InnoHK；同济大学；上海交通大学；浙江大学；南洋理工大学
- **预印本：** https://arxiv.org/abs/2610.11194
- **HTML：** https://arxiv.org/html/2610.11194v1
- **版本：** v1，2026-10-08，cs.RO / cs.CV
- **项目页 / 代码 / 数据：** 摘要页与 HTML 均未列出项目页、仓库或数据链接；截至 2026-10-12 未核实到任何公开发布。
- **核查日期：** 2026-10-12

## 论文要解决的问题

杂乱场景最能反映真实使用，却缺少大规模灵巧抓取训练数据；现有场景级抓取标注依赖慢速的场景级优化。同时，同一场景里一个物体可有多种可行抓取（多模态），标准扩散去噪损失会把多种模式平均掉，末端毫米级接触定位也不准。

## 方法与训练信息

**数据（seed-and-filter）**
1. 对每个物体用力闭合估计器优化 4,000 个候选抓取，通过仿真抬举测试的作为种子抓取。
2. 在 Isaac Gym 中把 2–7 个物体（桌面子集）或 2–10 个物体（箱子子集）落到支撑基座上；稳定判据为速度低于 0.03 m/s 与 0.10 rad/s 或 500 步；拒绝沉降/悬浮超过 3 mm 或物间穿插超过 2 mm 的场景。
3. 把种子抓取变换到场景位姿，保留与目标接触、无支撑面或非目标碰撞的抓取（物-支撑穿插容忍至 2 cm）。
4. 用 Blender 以五台 Intel RealSense D455 渲染 RGB、深度、语义分割与相机参数。
5. 规模：超过 2.6M 场景、0.4B 场景级抓取真值、超过 6,000 个可抓物体；基座 244 张桌子、79 个开口箱、21 个货架；物体来自 OmniObject3D、ShapeNetCore v2、Google Scanned Objects。子集：D_std 1.14M、D_mix 0.94M、D_box 0.14M、D_grid 0.28M、D_shelf 0.14M；训练/测试 8/2。

**模型**
- 输入：RGB（DINOv2 特征）、深度（轻量适配器反投影为 XYZ 图）、指令掩码、相机参数、桌面高度，融合为条件嵌入；多视角用学习的 alpha 图做早期 RGB 融合、深度重投影到锚相机、相机关系 token。
- 输出：扩散去噪器生成 Shadow Hand（24 DoF）抓取：腕平移、腕旋转、关节角。
- Soft Winner-Takes-All：M=64 个假设共享同一注入噪声，用温度 τ 和 log-sum-exp 聚合逐假设噪声误差。
- 可微手部仿真的物理损失：拇指接触、对向手指接触、对跖闭合（approach 向量余弦）、SDF 穿插。权重 λ_WTA=1、λ_phys=100，子项比 5:1:0.2:2；物理损失前 10 轮关闭，第 10–20 轮线性升到 100。
- 训练：AdamW，lr 1e-4，weight decay 1e-4，25 轮，NVIDIA A800。
- 推理：从一个条件嵌入采 64 个候选，按 S = S_kin − R_con + Ω_phy 排序（力闭合稳定性、额外指尖接触奖励、穿插/碰撞惩罚），取最高分有效抓取。

## 论文报告的结果

- 指标：SR_scene（真实场景重力下成功，主指标）、SR_any、SR_one、CFR（无碰撞率）；成功判定时移除支撑面。基线均在 OmniDex 训练集上重训：DexGraspVLA、DexGraspNet 2.0、Grasp-as-You-Say。
- D_std 单视角：OmniDex 已见物体 SR_scene 33.60、CFR 84.12，未见物体 40.15、84.61；多视角 37.35 / 45.05。基线（已见）：DexGraspVLA 24.30 / 81.44，DexGraspNet 2.0 28.03 / 81.06，Grasp-as-You-Say 0.24 / 38.57。正文称 OmniDex 比 DexGraspVLA 高 14.81 点，但正文与表 3 的 55.80 所在列不一致（带星号的手指施力变体列），数字读法有歧义。
- 其他子集（单视角，已见物体 SR_scene）：D_mix 32.05；D_box 32.34（CFR 99.12）；D_grid 55.01（CFR 95.46）；D_shelf 29.49（CFR 81.95）。
- 数据规模消融（20%→100%）：SR_scene 40.50、41.04、43.84、44.32、44.41。
- 假设数 M：1/4/16/64 对应 SR_scene 36.08/41.06/43.41/44.41，推理 219/266/544/1598 ms；200 步测试时优化 SR_scene 36.16、CFR 仅 2.56、6045 ms。物理排序把 SR_scene 从 30%（随机选）提到 40%。
- 失败分析：未成功的 95.71% 是接触后不稳，4.29% 接触不足；碰撞失败中 61.26% 撞支撑基座。
- 可达性：Franka 臂五种基座位置下 83.75%–87.98% 的成功抓取运动学可达。
- 真机：Tianji Marvin M6S 臂 + BrainCo Revo2 手，抓苹果、山竹、杯子，仅定性，未报告成功率。

## 复现与解释边界

- 基准构建与主评测均为仿真；sim-to-real 缺口（传感器噪声、标定误差、接触参数、执行不确定性）未量化。
- 数据生成与训练需要大量 GPU；最高 SR_scene 仍不到 50%（多视角消融 49.87）。
- 未发现任何代码或数据发布声明。

## Robotics_Notebooks 映射

- 详情节点：wiki/entities/paper-omnidex.md
- 汇总页：wiki/overview/frontier-manipulation-2026-10-05-10-09.md

## 一手来源

- arXiv 摘要：<https://arxiv.org/abs/2610.11194>
- arXiv HTML：<https://arxiv.org/html/2610.11194v1>
