# FASTER: Value-Guided Sampling for Fast RL（arXiv:2604.19730）

> 来源归档（paper）

- **论文：** <https://arxiv.org/abs/2604.19730>
- **项目页：** <https://pd-perry.github.io/faster/>
- **Robomimic 代码：** <https://github.com/alexanderswerdlow/faster>
- **π0.5 / VLA 代码：** <https://github.com/alexanderswerdlow/faster_vla>
- **作者 / 机构：** Perry Dong、Alexander Swerdlow、Dorsa Sadigh、Chelsea Finn；斯坦福大学（Stanford University）
- **一句话说明：** 把扩散去噪中的候选筛选建模为 MDP，用去噪 Q 函数提前淘汰低价值动作，在减少完整去噪采样成本的同时保留 best-of-N 收益。
- **入库日期：** 2026-10-07

## 核心摘录

1. **问题：** best-of-N 扩散策略需完整去噪多个候选再选一个，策略质量提升的推理开销很大。
2. **方法：** 将候选动作的扩散去噪过程表示为 MDP，学习策略/价值函数在去噪途中保留或丢弃候选；最终仅完整去噪幸存候选并执行。
3. **训练：** 去噪 critic 采用时序差分学习；终端价值与环境动作价值关联。
4. **结果：** 在长时程操纵 online 与 batch-online RL 中稳定提升底层策略；应用到预训练 VLA 后，论文报告同等性能且训练和推理计算显著下降。
5. **开放状态：** 项目页分别链接 Robomimic 通用代码和 π0.5 VLA 代码；VLA 仓库使用独立训练/环境虚拟环境，通过 UNIX socket 传递动作与观测。

## 对 wiki 的映射

- [paper-faster](../../wiki/entities/paper-faster.md)
- [EXPO](../../wiki/entities/paper-expo.md)
