# RMA 论文通读（特权训练视角）

> 来源归档（原文通读 · 支撑 `wiki/concepts/privileged-training.md`）

- **标题：** RMA: Rapid Motor Adaptation for Legged Robots
- **作者：** Ashish Kumar, Zipeng Fu, Deepak Pathak, Jitendra Malik
- **类型：** paper / quadruped locomotion / privileged training / sim2real
- **原文：** <https://arxiv.org/abs/2107.04034>（2026-11 通过 arXiv PDF v1 获取全文）
- **一句话说明：** 特权环境向量 $e_t$ 经编码器得 extrinsics $z_t$ 训练 base policy；再用状态-动作历史监督训练 adaptation module 估计 $\hat z_t$，部署无需特权信息。
- **核对范围：** 通读正文 §I–§VI（含相关工作、§III 方法、§IV 实验设置与训练细节、§V 仿真/室内/户外结果与适应分析、结论）及补充材料（指标、Algorithm 1/2、PPO 超参、额外适应分析、额外仿真泛化）；参考文献列表仅浏览，未逐条核对被引工作。
- **通读：** 已完成

## 核心摘录（均出自原文）

- **两阶段：** Phase 1 用 PPO 联合训练 $\pi$ 与环境因子编码器 $\mu$，$z_t=\mu(e_t)$，$a_t=\pi(x_t,a_{t-1},z_t)$；Phase 2 冻结二者，用 MSE 监督 $\phi$ 回归 $z_t$。
- **维度：** $x_t\in\mathbb{R}^{30}$（12 关节位置 + 12 关节速度 + roll/pitch + 4 足触地二值），$a_t\in\mathbb{R}^{12}$，$e_t\in\mathbb{R}^{17}$（质量及位置 3 维、电机强度 12 维、摩擦、局部地形高度），$z_t\in\mathbb{R}^{8}$。
- **特权量的实现细节：** 地形高度取四足下高度离散到小数一位后的最大值，使控制器不依赖对局部地形的快速精确感知。
- **为何蒸馏 extrinsics 而非 $e_t$：** 作者称 $z_t$ 是环境参数的低维非线性投影，缓解参数共变导致的不可辨识；不要求 $z_t$ 符合"真值"，只要导出正确动作。对照 SysID 基线（直接预测 $e_t$）表现更差（仿真成功率 56.5% vs RMA 73.5%）。
- **Adaptation module：** 输入过去 $k=50$ 步（0.5 s）状态-动作，2 层 MLP 嵌入到 32 维 + 3 层 1D CNN + 线性投影；on-policy 数据（用随机初始化 $\phi$ 的预测 $\hat z_t$ 驱动 $\pi$ 采样，类似 DAgger），因为专家轨迹只含"好"轨迹，对偏离不鲁棒。
- **异步部署：** $\phi$ 10 Hz，$\pi$ 100 Hz；作者称该设计对 A1 有限机载算力关键；直接把历史喂给 $\pi$ 的单模块方案在仿真中步态不自然、性能差，且板载只能跑 10 Hz。
- **训练开销：** Phase 1 约 24 小时 / 12 亿步（15000 次迭代 × 80000 batch，单 GPU 桌面机），Phase 2 约 3 小时 / 8000 万步（1000 次迭代）。RaiSim 仿真。
- **奖励与课程：** 10 项生物能量启发奖励（前进上限 0.35 m/s、功、地面冲击、足滑等）；惩罚项系数先小后指数增大（$k_{t+1}=k_t^{0.997}$，$k_0=0.03$），否则策略学会原地不动/摔倒；质量、摩擦、电机强度扰动难度线性增加。
- **仿真对比（Table II）：** RMA 73.5% 成功率，Expert（用真值 $z_t$）76.2%，Robust(DR) 62.4%，SysID 56.5%，AWR 41.7%（需 40k 样本），RMA w/o Adapt 52.1%。
- **实机（A1）：** 5 次试验；载荷至 12 kg 可走；油滑面 90% 成功；户外沙、泥、草丛 100%、楼梯下行 70%。A1 默认控制器在泡沫上 20% 成功。
- **局限（作者所述）：** 仅本体感知（盲），大扰动（下楼突然跌落、多腿被岩石卡住）会失败；需外感知（视觉）。
