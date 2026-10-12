# Atomic Motion Primitives 遥操作论文来源归档

## 元数据

- **题目：** Beyond Retargeting: Low-Latency and Robust Humanoid Whole-Body Teleoperation with Learned Atomic Motion Primitives
- **方法：** 特权稀疏 MoE Teacher + 因子化类别码本 Student（学习式原子动作基元，免显式重定向）
- **作者：** Xiayan Xu, Jiyu Yu, Xingzhou Chen, Siyi Qian, Zongyu Ma, Lilu Liu, Ling Shi, Haodong Zhang（通讯）
- **单位：** 香港科技大学（HKUST）；浙江大学；腾讯 Robotics X；湖南大学
- **预印本：** https://arxiv.org/abs/2610.07891
- **HTML：** https://arxiv.org/html/2610.07891v1
- **版本：** v1，2026-10-06，cs.RO
- **项目页 / 代码：** arXiv 摘要页与正文均未给出项目页或 GitHub 链接；截至 2026-10-12 核查未发现官方代码或权重（入库日期 2026-10-12）。

## 论文要解决的问题

常规 VR / 动捕全身遥操作先把人体动作重定向（retarget）成机器人参考，再交给低层跟踪策略。重定向本身在作者的共享仿真运行时中要 21.9–22.6 ms，且会产生机器人不可行的目标；输入一旦出现遮挡、噪声或 OOD 动作，跟踪策略也容易失稳。论文要把「人体描述符 → 关节指令」做成一次前向，并让策略对残缺、噪声输入有可恢复的先验。

## 方法与训练信息

1. **Stage 1 Teacher：** 稀疏 MoE actor（8 个细粒度专家 + 1 个共享专家，top-k=2 路由），PPO 训练（lr 1e-3，γ=0.99，GAE λ=0.95，clip 0.2，entropy 5e-3，horizon 24，负载均衡权重 0.025）。观测为本体感知、机器人坐标系参考（带短前瞻）及特权根速度/足接触；重定向参考只在训练期使用。自适应采样器按 1 s 分段、失败率 EMA 加权。
2. **Stage 2 Student：** 输入人体描述符（根 13 维、J 个关键点 3J 维、双腕朝向 12 维）与 L 帧本体感知历史；训练时加 Bernoulli 关键点可见性掩码与均匀噪声。因子化类别码本 C=128 通道 × D=12 项，Group-wise straight-through Gumbel-Softmax；编码器看干净描述符，估计器看被污染描述符并匹配编码器码字（stop-gradient）。损失为对 Teacher 动作均值的 Huber 行为克隆（δ=0.7）+ 码本损失（λ=0.3）+ 5 个辅助头。
3. **执行：** 动作 u 映射为关节目标 q0+α⊙u（α∈[0.07,0.55]），50 Hz；部署时仅运行估计器、量化器、解码器和策略。
4. **数据与平台：** AMASS / LAFAN1 / OMOMO 重定向库，14,147 训练片段 / 4,073 测试片段；IsaacSim 训练；MjLab 上用 6,495 条 SONIC bones-seed 轨迹做跨仿真评测；机器人 Unitree G1。并行环境数原文未说明。

## 论文报告的结果

- **σ=0.1 m 关键点噪声跟踪（Table I）：** 码本 Student SR 0.9590，MPJVE 1.1698；无码本 MLP SR 0.9529、MPJVE 1.4412；TWIST 0.8451、TWIST2 0.6199、GMT 0.7540、CLONE 0.8335、ANY2TRACK 0.0556。七项指标中码本 Student 六项最优，MLP 的 MPKPE 略好（0.2060 vs 0.2159）。
- **仿真端到端指令延迟（Table II，RTX 4060，FP32）：** 码本 0.604 ms；CLONE 0.933 ms；TWIST 22.372 ms；TWIST2 23.346 ms；GMT 1923.050 ms（含 1.9 s 前瞻）；SONIC(G1) 204.077 ms。此数不含感知、通信与执行器延迟。
- **真机延迟：** 55 s 出拳跟踪、光流对齐 200 Hz 采样，中位滞后 147.5 ms（含感知、通信、控制、执行器）；仅此一次试验，未测基线。
- **OOD 成功率（Table III）：** 关键点噪声 0.30 m：码本 0.9487，MLP 0.8675，TWIST 0.6344；未见动作（Motion-X 子集）：码本 0.6067，MLP 0.5433，TWIST 0.4467；根高度漂移 0.20 m：码本 0.8572，CLONE 0.0000。
- **Teacher 消融（Table IV）：** 完整 98.88 SR；去 MoE 90.23；去自适应采样 97.80；二者皆去 82.13。
- **码本容量（Table V）：** SR 在 0.932–0.943 之间，作者按码字使用健康度选 C=128、D=12。
- **关键点数：** SR 随 k 从 3（头+双腕）到 13 由 0.9153 升到 0.9366。
- **真机：** 同一 checkpoint 无微调，由 VR、光学动捕、单目视频（GVHMR）、文本生成动作（HY-Motion 1.0）驱动 G1；仅定性展示，无成功率。

## 复现与解释边界

- 无公开代码、权重或项目页，复现依赖作者自述的超参；并行环境数、部分超参含义原文未给。
- 延迟 0.604 ms 只是策略前向；真机端到端 147.5 ms 才含全链路，两数不可混比。
- 真机四种输入源均只有定性结果，OOD 数字来自注入扰动。
- 码本为逐帧，不显式建模长时运动结构（作者列为未来工作）。「首个免重定向遥操作策略」为作者自述，未经本页核实。

## Robotics_Notebooks 映射

- 详情节点：wiki/entities/paper-atomic-motion-primitives-teleoperation.md
- 任务入口：wiki/tasks/teleoperation.md
- 汇总页：wiki/overview/frontier-humanoid-quadruped-2026-10-05-10-09.md

## 一手来源

- arXiv 摘要：<https://arxiv.org/abs/2610.07891>
- arXiv HTML 全文：<https://arxiv.org/html/2610.07891v1>
