# Chronos Project Page

> 来源归档

- **标题：** Chronos — Physics-Informed Full-History Framework for Non-Markovian Long-Horizon Manipulation
- **类型：** site / project page
- **URL：** <https://chronos-manipulation.github.io/>
- **论文：** <https://arxiv.org/abs/2606.30318>
- **代码仓：** <https://github.com/yulinzhouZYL/Chronos>
- **权重：** <https://huggingface.co/yulinzhouZYL/Chronos-RMBench>
- **机构：** 华中科技大学（HUST）
- **入库日期：** 2026-07-27
- **一句话说明：** 官方项目页：方法三阶段概览、RMBench / 参数量对比数字、ALOHA 与真机双臂对比视频、BibTeX。

## 开源状态（项目页核查，2026-07-27）

| 项 | 状态 |
|----|------|
| Paper | arXiv **2606.30318** |
| Code / Resources | 链到 GitHub `yulinzhouZYL/Chronos` |
| 仓内实现 | RMBench 策略 + 真机 UR3 采数/训练/推理；HF ckpt |
| Coming soon（README） | 清理后的 ALOHA、RoboTwin 2.0 代码 |
| 结论 | **已开源（部分）** — 主复现路径（RMBench + 真机）可用 |

## 页面结构（策展）

| 区块 | 内容要点 |
|------|----------|
| Hero 数字 | RMBench **73.6%**；相对 π₀.₅ **+62.4**；**10×** 更少参数；真机平均 **78%** |
| Overview | 历史作策略潜状态；SSM 全因果历史；IMLE 多模态先验；二阶加速度桥 |
| Framework | Full-History State Encoding → IMLE Action Prior → Second-Order Action Bridge |
| Key Results | 相对 Mem-0 **+22.8 pt**、**30×** 更少参数；真机记忆依赖 **72%** |
| Simulation Gallery | ALOHA 插入：diffusion 头 vs Chronos 二阶桥运动对比 |
| Real-World | π₀.₅ vs Chronos：长序列 / 可见操作 / 记忆依赖扩展 |
| BibTeX | `zhou2026chronos` |

## 对 wiki 的映射

- 论文：[`sources/papers/chronos_arxiv_2606_30318.md`](../papers/chronos_arxiv_2606_30318.md)
- 代码：[`sources/repos/chronos.md`](../repos/chronos.md)
- 沉淀 **[`wiki/entities/paper-chronos.md`](../../wiki/entities/paper-chronos.md)**
