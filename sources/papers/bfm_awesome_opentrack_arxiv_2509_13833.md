# Track Any Motions under Any Disturbances（arXiv:2509.13833）

> 来源归档（ingest · 原文通读）

- **标题：** Track Any Motions under Any Disturbances
- **类型：** paper
- **BFM 分类：** 02 Goal-conditioned 学习（[awesome-bfm-papers](https://github.com/friedrichyuan/awesome-bfm-papers)）
- **出处：** 2025 · arXiv:2509.13833v3（2025-09-30）
- **作者：** Zhikai Zhang*, Jun Guo*, Chao Chen, Jilong Wang, Chenghuai Lin, Yunrui Lian, Han Xue, Zhenrong Wang, Maoqi Liu, Jiangran Lyu, Huaping Liu, He Wang, Li Yi:
- **机构：** Tsinghua University；Peking University；Galbot；Shanghai Qi Zhi Institute
- **论文链接：** <https://arxiv.org/abs/2509.13833>（PDF：<https://arxiv.org/pdf/2509.13833>）
- **项目页：** <https://zzk273.github.io/Any2Track/>
- **代码：** <https://github.com/GalaxyGeneralRobotics/OpenTrack>
- **入库日期：** 2026-05-26
- **核查日期：** 2026-11-21
- **一句话说明：** 提出两阶段 RL 框架 Any2Track（AnyTracker + AnyAdapter），将“通用动作跟踪”与“在线动力学适应”解耦，使 Unitree G1 在地形起伏、外力推拉与质量/质心/摩擦变化下仍能零样本完成高动态与接触丰富动作跟踪。
- **核对范围：** 通读 arXiv:2509.13833v3 全文（§I Introduction、§II Related Works 与 Table I、§III Any2Track 方法与 Tables II–III / Fig. 2、§IV Experiments 仿真与 G1 真机评测及 Tables IV–VI / Figs. 3–4、§V Conclusions）。
- **通读：** 已完成

## 核心技术与实验事实（原文核对）

- **Stage 1 — AnyTracker（§III-A）：**
  - 在 **MuJoCo**（8 GPU 并行 PPO）上无动力学扰动训练基础通用 tracker，观测包含角速度、重力投影、关节位置/速度、上一帧动作 $s_t$ 与下一帧局部系参考目标 $g_t$。
  - 保留 **LAFAN1 + AMASS** 中的高动态与接触丰富动作（仅按 PHC 滤除依赖不可用楼梯等场景的动作）。
  - **规范化动作空间（CAS）：** 输出相对于下一帧参考关节位置 $\tilde{q}_{t+1}$ 的残差 PD 目标：$q_d = \tilde{q}_{t+1} + \alpha \tanh(\pi(a_t|s_t))$，通过逐关节缩放 $\alpha$ 缓解自由度异质优化困难。
  - **运动聚类与 Specialist-to-Generalist：** LAFAN1 按原类别分簇；AMASS 用 HumanML3D 的 VERB 文本标签经 **CLIP** 嵌入后做 **K-Means（6 簇）**，分别训练专家策略后用 **DAgger** 蒸馏为通才策略。
- **Stage 2 — AnyAdapter（§III-B）：**
  - 引入三类环境扰动（Table III）：Perlin 随机地形（高度 0.3 m、地面摩擦 $U(0.3,2.0)$）、外力推拉（间隔 $U(5,10)$、速度幅值 $U(0.1,1.0)$）、物理属性变化（躯干质量 $U(-3,6)$ kg、质心偏移 $U(-0.15,0.15)$、关节摩擦/转子惯量缩放）。
  - **动力学世界模型代理任务：** 历史编码器 $\phi$ 读取 $H=79$ 步 $(s,a)$ 序列提取嵌入 $e_t=\phi(h_t)$，世界模型 $\omega$ 自回归预测未来 $N=20$ 步状态 $\hat{s}_{t+i+1}=\omega(\hat{s}_{t+i}, a_{t+i}, e_{t+i})$，以 $L_1$ 损失 $L_{wm}$ 联合更新。
  - **零初始化逐层 Adapter：** 冻结 AnyTracker 参数，引入逐层零初始化适配器 $\xi=\{\xi_1,\dots,\xi_M\}$ 做特征融合并交替更新 adapter 与 history encoder，避免直接微调破坏原有表现力。
- **实验结果（§IV，29-DoF Unitree G1）：**
  - Table IV（AMASS-test + LAFAN1 无扰动）：AnyTracker SR **89.23%**、MPJPE **27.96 mm**、MPJVE **6.43 mm/frame**，优于复现的 Exbody2（79.68% / 34.70 / 13.69）、OmniH2O（75.64% / 36.12 / 12.24），且去除 CAS 或蒸馏分别降至 84.92% 与 83.32%。
  - Table V（LAFAN1 在线扰动适应）：在无扰动 / 地形 / 外力 / 物理变化下 SR 分别达 **89.8% / 83.2% / 59.0% / 80.6%**，全面超过 PPO（87.4/73.0/50.0/73.6%）、RMA Stage 1/2、DWL、Dual History；去掉 World Model 在外力与物理变化下分别跌至 49.8% 与 70.4%。
  - Table VI（Unitree G1 真机，5 条轨迹均值）：复杂地形（木板/纸板/泡沫/织物）MPJPE **18.34 mm**（vs PPO+DR 37.21 mm）、外部绳索牵拉约束 **19.17 mm**（vs 39.84 mm）、背部 **5 kg** 负重 **23.24 mm**（vs 37.52 mm）。

## 对 wiki 的映射

- [OpenTrack / Any2Track](../../wiki/entities/paper-opentrack.md)
- [any2track](../../wiki/methods/any2track.md)
- [bfm-41-papers-technology-map](../../wiki/overview/bfm-41-papers-technology-map.md)
