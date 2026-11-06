# Reinforcement Learning with Random Delays 论文通读（控制环路延迟视角）

> 来源归档（原文通读 · 支撑 `wiki/formalizations/control-loop-latency-modeling.md`）

- **标题：** Reinforcement Learning with Random Delays
- **作者：** Simon Ramstedt, Yann Bouteiller, Giovanni Beltrame, Christopher Pal, Jonathan Binas
- **类型：** paper / rl / delay / real-time-control
- **原文：** <https://arxiv.org/abs/2010.02966>（arXiv PDF v3，2021-05，pdftotext 取得全文）
- **一句话说明：** 把「观测延迟 + 动作延迟」写成随机延迟 MDP（RDMDP），指出必须用最近 K 个已发出动作扩充状态才能恢复 Markov 性，并据此提出离策略多步价值估计的 DCAC 算法。
- **核对范围：** 通读摘要、§1–§7 正文、附录 A–G（实践考虑、实现细节与超参、可视例、补充实验、定义与引理、定理证明）；附录 F/G 的证明仅核对结构与结论，未逐步重推每一步代数；未阅读其引用的 Walsh 2008、Ramstedt & Pal 2019 等原文。
- **通读：** 已完成

## 核心摘录（均出自原文）

- **延迟分解（§2、图 1）：** 延迟环境 = 无延迟环境 + 延迟的通信动力学。观测延迟 ω 指观测采集结束到被用于计算动作的步数；动作延迟 α 指影响该观测的最后一个动作开始计算到观测采集结束前一步的步数；总延迟 ω+α。实数延迟向上取整到整数步。
- **Markov 性（§2、附录 D.1）：** 任何超过一步的延迟都使「只用最新收到的观测」违反 Markov 假设；需把至少最近 K 个已发动作（K 为观测与动作延迟之和的最大值）并入状态。仅一步常数动作延迟时，朴素 SAC 就已接近随机表现。
- **RDMDP（定义 1–2）：** 状态 X = S × A^K × N²（延迟观测、动作缓冲、ω、α）；常数延迟时退化为 Walsh 等的 CDMDP，ω、α 可从状态中去掉。把延迟放进状态可建模自相关延迟（丢弃过期信息）并支持显式信用分配。
- **部分轨迹重采样（§3、定理 1）：** 当 ω_t+α_t ≥ t 时，动作缓冲中的新近动作尚未影响后续观测，可按当前策略重采样，使离策略子轨迹变为同策略子轨迹；无需重要性权重，因此避免了方差爆炸（对比 Retrace）。n 步估计偏差按 γ^n 缩小（引理 1）。
- **DCAC（§4）：** 基于 SAC，critic 估计状态价值 v 而非 q，目标为延迟同策略 n 步软价值；actor 损失偏差为 SAC 的 E[γ^n] 倍（命题 1），需对动作缓冲做时间反传。ω=0、α=1 时退化为 RTAC。
- **实验（§5、图 6–7、附录 D）：** 延迟版 MuJoCo；常数延迟 ω=2、α=3 时 DCAC 优势显著；延迟采样自一份真实飞行机器人 WiFi 数据集（时间步 20 ms，数据集含至 1 s 的离群值，80 ms 以上的延迟几乎总被更新信息取代故可截断）。Ant 对所有算法都过难。6 个种子，90% 置信区间。
- **实践（附录 A）：** 用时间戳实现“只保留最新信息”；测量延迟需动作与观测时间戳（时钟同步时可直接由时间戳求 ω）；最大延迟是设计选择，超过 K 的观测应丢弃；观测采集非瞬时或观测由多部分拼成时，动作缓冲要相应加长。
- **实现（附录 B）：** actor / critic 各为 2×256 MLP，Adam 3e-4，γ=0.99，批 128，τ=0.005，回放池 1e6，双 critic。

## 对 wiki 的映射

- [`wiki/formalizations/control-loop-latency-modeling.md`](../../wiki/formalizations/control-loop-latency-modeling.md)

## 参考来源

- 论文：<https://arxiv.org/abs/2010.02966>
