# MBPO 论文通读（模型使用与短 rollout 视角）

> 来源归档（原文通读 · 支撑 `wiki/methods/model-based-rl.md`）

- **标题：** When to Trust Your Model: Model-Based Policy Optimization
- **作者：** Michael Janner, Justin Fu, Marvin Zhang, Sergey Levine（UC Berkeley，NeurIPS 2019）
- **类型：** paper / rl / model-based
- **原文：** <https://arxiv.org/abs/1906.08253>（2026-11 通过 arXiv PDF v3 获取全文）
- **一句话说明：** 从理论与实验两面研究"模型该怎么用"：证明带模型误差的回报下界，指出只靠悲观界会得出"别用模型"，再用经验测得的模型泛化率说明短的、从真实数据分支出去的 rollout 值得用，据此得到 MBPO（集成模型 + SAC）。
- **核对范围：** 通读正文 §1–§7（含 §4 三个定理、§5 算法 2、§6 全部实验与消融）、附录 A（定理 A.1–A.3 证明思路）、附录 B（引理）、附录 C（超参表 1）；证明的逐步代数推导只核对结构与结论，未逐行验算。未读 PETS、SAC、STEVE 等被引论文，相关内容仅依赖 wiki 既有页面。
- **通读：** 已完成

## 核心摘录（均出自原文）

- **问题（§1）：** 模型精度是策略质量的瓶颈，常使 MBRL 渐近性能不如 model-free；数据生成的便利要与模型数据的偏差权衡。
- **单调改进界（§4.1，定理 4.1）：** η[π] ≥ η̂[π] − C(ε_m, ε_π)，C 含 2γ r_max(ε_m+2ε_π)/(1−γ)² + 4 r_max ε_π/(1−γ)；ε_m 为模型泛化误差（可用验证损失估计），ε_π 为策略间 TV 距离（实践中测 KL，经 Pinsker 关联）。完整模型 rollout 使误差随有效视野 1/(1−γ) 二次放大。
- **分支 rollout（§4.2，定理 4.2）：** 从旧策略状态分布出发、在模型里按当前策略走 k 步；Dyna 是 k=1 的特例。但按保守常数取下界在 k=0（完全不用模型）最大，即该界无法证明模型有用。
- **经验泛化（§4.3，图 1，定理 4.3）：** 在 Hopper / Walker2d 上测得，模型误差随策略偏移增大，但训练数据越多这一增长率越小；用线性近似 ε_m' ≈ ε_m + ε_π·dε_m'/dε_π 后，模型误差足够低时最优 k* > 0，从而支持"有限、非零长度"的模型 rollout。
- **算法（§5，算法 2）：** B 个概率神经网络集成（高斯、对角协方差，bootstrap 捕获认知不确定性、概率输出捕获偶然不确定性），每步随机选一个成员预测；策略用 SAC；模型 rollout 从 D_env 中均匀采样的真实状态出发走 k 步，存入 D_model，再做 G 次策略更新。短 rollout 产生大量模型样本，使每个环境步可做 20–40 次梯度更新（model-free 通常不稳定）。
- **基准结果（§6.1，图 2）：** 标准 1000 步 MuJoCo（InvertedPendulum / Hopper / Walker2d / Ant / HalfCheetah / Humanoid），对 SAC、PPO、PETS、STEVE、SLBO，5 次试验。MBPO 渐近性能接近最佳 model-free，样本效率提升可达一个数量级（Ant 上 30 万步≈SAC 300 万步）；Hopper、Walker2d 分别约等于实时仿真 14、40 分钟；在 Ant 等高维任务上纯模型法 PETS 表现不佳。
- **消融（§6.2，图 3）：** (1) 无模型但同样高 G 的 SAC 只略加速，追不上 MBPO；(2) Hopper 上最优 k 从 1 线性增至 15，固定 k=1 仍保留大部分收益，200 步 rollout 模型尚准但策略训练更差，500 步太不准；(3) 模型价值展开（H 步 Q 目标，MVE/STEVE 类）中 H=1 亦出奇有效，与 MBPO 的结合在初步实验中无额外提升；
- **模型利用（图 4）：** Hopper 上策略在模型与真实环境中的回报高度相关，说明短 rollout 下策略没有利用模型漏洞；多数环境中模型回报低估真实回报。长 rollout 的预测不确定性增长、周期运动退化。
- **超参（附录 C，表 1）：** E=1000 环境步/epoch，M=400 模型 rollout/环境步，B=7，网络为 4 层 MLP（200 或 400 宽），G=20–40；k 随 epoch 线性上调（如 Hopper 1→15、Walker2d/Ant/Humanoid 1→25 等）。
- **讨论（§7）：** 模型精度不再是硬瓶颈的关键是"限制用法"（短分支 rollout），而非追求完美长程模型。

## 对 wiki 的映射

- [`wiki/methods/model-based-rl.md`](../../wiki/methods/model-based-rl.md)
- [`wiki/methods/mbpo.md`](../../wiki/methods/mbpo.md)
