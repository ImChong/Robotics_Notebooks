# Qubit-to-Motion（arXiv:2611.11580）

> 来源归档（paper）

- **标题：** Qubit-to-Motion: Traceable Action Provenance in Hybrid QRL for Robotic Bin-Packing Sequencing
- **类型：** paper
- **作者与机构：** Hassen Nigatu（浙江大学机器人研究院，宁波）、Howard Li（University of New Brunswick，加拿大）
- **arXiv：** <https://arxiv.org/abs/2611.11580>
- **PDF：** <https://arxiv.org/pdf/2611.11580v1>（v1，2026-11-08，cs.RO）
- **代码（匿名评审仓库，非 GitHub）：** <https://anonymous.4open.science/r/Qubit-to-Motion-Quantum-Reinforcement-Learning-F7DC>
- **入库日期：** 2026-11-11
- **一句话说明：** 把四个 Pauli-Z 读数驱动的混合量子 RL 装箱顺序决策，用决策账本、路线封存与 SHA-256 指纹，一路追溯到 Panda 路点、关节运动与 MuJoCo 接触回放（仅仿真）。
- **核对范围：** 用 pdftotext 通读 arXiv v1 PDF 全文（摘要、I-VII 节、表 I-III、参考文献列表），无附录 / 补充材料（该版本未附）；图 1-5 只读了图注与表中数据，未逐图看图像；匿名仓库只读了 README 与文件列表（经站点 API），未读源码，未运行任何代码。
- **通读：** 已完成

## 开源状态

- 摘要称实现在匿名仓库（审稿期匿名，作者身份以论文页为准）。README（站点 API 读取）称包含 Python 源码、五个场景定义（seed 41/73/109/151/197）、1920x1080 MP4、检查点、测试与 `SHA256SUMS.txt`。
- 注：README 提到「bilateral fingertip-contact grasp model」，论文 III-F 写的是「运动学规定的 Panda 关节运动 + 传输阶段确定性吸附物体」，两者关系未核实。

## 核心摘录

1. 状态 / 动作：30 个物体、有向无环支撑图，动作只能选当前可行集合 A(s_k)（III-A，式 1-3）。
2. 网络：750-96-4 编码器 → 两层 4 比特 VQC（解析态矢量模拟，CNOT 环）→ 四个 Pauli-Z 期望 → 4-64-30 actor 与 4-64-1 critic；总可训练参数 75,163（图 2 图注；III-B）。
3. 训练：540 条贪心演示决策（18 个场景）BC 15 epoch，再 2 次 PPO 更新，每次 120 决策、3 遍优化（III-C，表 I）。
4. 评估：1 个检查点（seed 2025）× 5 个确定性 30 物体场景变体；150 次决策、1050 个路点、150 次放置全部通过（V，表 II）。
5. 全零读数干预改变 135/150 = 90.0% 决策（非强制决策 135/145 ≈ 93.1%）（V-B，表 II、III）。

## 对 wiki 的映射

- [paper-qubit-to-motion](../../wiki/entities/paper-qubit-to-motion.md)
