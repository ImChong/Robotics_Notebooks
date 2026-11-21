# LocoFormer（CoRL 2025）

> 来源归档（paper / CoRL 2025 Best Paper Award Finalist）

- **标题：** LocoFormer: Generalist Locomotion via Long-context Adaptation
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2509.23745>（v1 提交 2025-09-28，cs.RO，CC BY 4.0）
- **HTML：** <https://arxiv.org/html/2509.23745v1>
- **PDF：** <https://arxiv.org/pdf/2509.23745>
- **项目页：** <https://generalist-locomotion.github.io/>（标注 CoRL 2025 Best Paper Award Finalist）
- **公司博客：** <https://www.skild.ai/blogs/omni-bodied>（2025-09-24）
- **作者与机构：** Min Liu、Deepak Pathak、Ananye Agarwal / Skild AI
- **Tech Blog 引用：** <https://www.lightorigins.com/blog/light-react>
- **入库日期：** 2026-09-21
- **再核日期：** 2026-11-21
- **核对范围：** 通读 `https://arxiv.org/html/2509.23745v1` 全文（含摘要、§1 引言、§2 方法 2.1–2.2、§3 仿真与真机实验 3.1–3.2 与 Table 1、§4 相关工作、§5 结论、§6 局限及 Appendix A–C）与官方项目页 `https://generalist-locomotion.github.io/`；官方未开源代码/权重，未运行策略
- **通读：** 已完成
- **一句话说明：** 大规模 PPO + 程序生成机器人（100k 形态）+ Transformer-XL 跨 episode 记忆；在未见双足/四足/轮足形态及断腿/锁膝/锁轮等极端故障下实现秒级或跨试次 test-time 上下文自适应。

## 开源状态

- 历史记录：2026-09-21 曾写“待发布”。2026-10-07 与 2026-11-21 打开官方项目页复核：均仅提供 Paper / arXiv / Video / Blog 链接，未见官方代码、权重或数据下载入口（社区有非官方 WIP `lucidrains/locoformer`，不能当官方实现）。
- [官方时间线核查](../sites/skild-ai-timeline-audit-2026-10-07.md)

## 核心摘录（通读 arXiv:2509.23745v1 全文）

- **问题与直觉（§1）：** 传统运动策略在单一本体与窄域随机化下训练，上下文窗口仅几百毫秒，面对大范围形态变化或电机/肢体故障会失效。LocoFormer 将两个要素放大数个数量级：① 在 **程序化生成的 100k 机器人**（双足、四足、轮式双足、轮式四足）上施加极端域随机化与大规模 PPO；② 将策略上下文延长到 **跨越 trial 边界的多 episode 尺度**。
- **方法（§2）：**
  - **多 trial 适应公式（§2.1）：** 一个 episode 包含 $k$ 个 trial（训练中采样适应时间预算 $u \sim \text{Uniform}(0, U)$），记忆跨 trial 保留，目标函数最大化整个 episode 累积折扣回报；采用统一关节超集空间（unified joint space）拼接观测与目标关节位置。训练分两阶段：先短 trial 与小 $U$ 学快速适应，再拉长 trial 与 $U$ 准备真机部署。
  - **Transformer-XL 与推理加速（§2.2）：** 将长序列切分为固定长度段 $L$，当前段对前一段缓存隐藏状态做 stop-gradient 拼接 $\tilde{\mathbf{h}}_{z+1}^{n-1}=[\text{SG}(\mathbf{h}_z^{n-1})\circ \mathbf{h}_{z+1}^{n-1}]$，有效记忆随层数线性增长 $O(NL)$（6 层 TXL、$L=128$ 可达 896 步，在 50 Hz 下约 18 s）。真机部署采用 KV-cache 维持最近 $2L-1$ 步，把推理复杂度降为线性。
- **仿真定量结果（§3.1, Table 1, 10 款未见机器人 × 1,000 崎岖地形随机环境，归一化位移 $[0,1]$）：**
  - **Ours (zero-shot)** 均分 **0.96±0.19**（G1 0.96, H1 0.98, GR1 0.95, TRON1 0.87, BK-H 0.98, A1 0.92, Spot 1.00, AnyMal C 0.95, TRON1-W 0.99, Go2-W 0.97）。
  - **Ours (few-shot, 5 s 适应窗)** 均分 **0.98±0.13**（TRON1 从 0.87 升至 **0.96**，提升约 10%；逼近单本体 Expert Policy 上限 0.99±0.07）。
  - 对照基线：**GRU** 仅 **0.37±0.41**；**Conditioning**（特权运动学输入但上下文仅 64 帧）为 **0.78±0.37**（在无踝关节双足 TRON1 上仅 0.07）。
  - **表征演化（§3.1, Fig. 4b）：** 4,096 次零样本 rollout 中第 2 层表征在前 1 秒高度相似，5 秒内按本体分化出清晰聚类，说明模型在线隐式完成本体辨识。
- **真机零样本与极端故障适应（§3.2, Fig. 5）：** 部署于 Unitree G1、H1、Go2、Go2-W（含 Go2/Go2-W 仅后腿双足模式）：A/E 锁死膝关节后 2–3 s 重建三足重心步态；B 无踝双足前两次跌倒后利用 TXL 缓存第 3 次稳定行走；C Go2-W 软件锁死轮毂增益后切为步行，解锁后切回节能滚动，外加负载也可平衡；D 腿部加高跷重调步频落足；F 截短小腿去掉 4 DoF 后 7–8 s 学会大腿大幅摆动步行。
- **计算成本与局限（§6, Appendix C）：** 跨 100k 机器人训练算力约为单本体专家策略的 500 倍（摊销到每个机器人从 1 天降至 0.005 天；6 层 TXL 4k 迭代约 28 h）；局限在于算力开销大，且程序化形态空间为手工设计。

## 对 wiki 的映射

- [paper-locoformer](../../wiki/entities/paper-locoformer.md)
- [3 篇技术地图](../../wiki/overview/lightorigins-3blogs-technology-map.md)
- 引用来源上下文：[lightorigins_light_react_2026-09-09.md](../blogs/lightorigins_light_react_2026-09-09.md)
