# Open X-Embodiment 论文通读（具身数据集全景页视角）

> 来源归档（原文通读 · 支撑 `wiki/overview/embodied-dataset-panorama-2026-10.md`）

- **标题：** Open X-Embodiment: Robotic Learning Datasets and RT-X Models
- **作者：** Open X-Embodiment Collaboration（21 家机构，数百位作者）
- **类型：** paper / dataset / cross-embodiment
- **原文：** <https://arxiv.org/abs/2310.08864>（2026-11 通过 arXiv PDF v9 获取全文）
- **一句话说明：** 把 60 个已有机器人数据集统一成 RLDS 格式，得到 22 种本体、100 万+ 条真机轨迹的 OXE 数据集，并用 RT-1 / RT-2 训练出 RT-X，验证跨本体正迁移。
- **核对范围：** 通读摘要、§I 引言、§II 相关工作、§III 数据集与分析、§IV RT-X 设计、§V 全部实验（表 I、表 II、图 4–5）与 §VI 讨论；该版本无独立附录正文，参考文献仅浏览。未复核各子数据集原论文，也未阅读被全景页引用的其余 84 项数据集原文及公众号综述全文（公众号页面需人机验证，无法抓取）。
- **通读：** 已完成

## 核心摘录（均出自原文）

- **规模（§III-A）：** 汇集 60 个已有数据集、34 个实验室，转成统一格式，含 22 种本体（单臂、双臂、四足），100 万+ 真机轨迹；摘要称 527 种技能（160266 个任务）。
- **格式（§III-A）：** 采用 RLDS，序列化为 tfrecord，容纳不同数量的 RGB / 深度相机与点云，支持主流深度学习框架并行加载。
- **分析（§III-B）：** Franka 是最常见本体；技能以 pick-place 为主，长尾含擦拭、装配；用 PaLM 从语言指令中抽取物体与行为。
- **数据对齐（§IV-A）：** 仅粗对齐：每个数据集选一个典型相机视角，动作转成 7 维末端执行器向量（含夹爪）；不对齐坐标系，动作可为绝对 / 相对位置或速度，因此同一动作向量在不同机器人上动作不同。
- **模型（§IV-B）：** RT-1（35M 参数，EfficientNet + FiLM + Transformer）与 RT-2（VLM 微调，动作作为文本 token）；动作离散为 256 档。
- **训练混合（§IV-C）：** 实验只用其中 9 种本体的数据（当时的全部数据，之后数据集继续扩展）；RT-2-X 为 VLM 数据与机器人数据约 1:1 联合微调。
- **结果（§V）：** 共 3600 次评测、6 台机器人。小数据域上 RT-1-X 在 5 个中的 4 个胜过各数据集原方法，平均成功率高约 50%；大数据域（Bridge、RT-1 数据）RT-1-X 欠拟合，55B 的 RT-2-X 才优于原方法，说明正迁移需要足够容量。
- **涌现技能：** Google Robot 上 RT-2-X 比 RT-2 约 3 倍；去掉 Bridge 数据后显著下降，指向来自 WidowX 数据的迁移。
- **消融（表 II）：** 图像历史、Web 预训练、55B 对 5B 都有明显作用；联合微调与微调表现相近（作者归因于数据更多样）。
- **局限（§VI）：** 未覆盖感知与执行形态差异很大的机器人，未研究对新机器人的泛化，也未给出何时发生正迁移的判据。

## 对 wiki 的映射

- [`wiki/overview/embodied-dataset-panorama-2026-10.md`](../../wiki/overview/embodied-dataset-panorama-2026-10.md)
- [`wiki/concepts/open-x-embodiment.md`](../../wiki/concepts/open-x-embodiment.md)

## 参考来源

- 论文：<https://arxiv.org/abs/2310.08864>
