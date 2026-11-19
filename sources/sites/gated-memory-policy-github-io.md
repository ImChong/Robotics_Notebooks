# Gated Memory Policy 项目页

- **论文：** [arXiv:2604.18933](https://arxiv.org/abs/2604.18933)
- **项目页：** <https://gated-memory-policy.github.io/>
- **代码：** <https://github.com/real-stanford/gated-memory-policy>
- **权重与数据：** <https://huggingface.co/datasets/yihuai-gao/gated-memory-policy> / <https://huggingface.co/yihuai-gao/gated-memory-policy/tree/main>
- **机构：** Stanford University（RSS 2026）
- **入库日期：** 2026-11-19
- **核对范围：** 通读项目页全部内容（摘要、MemMimic T1–T6 真机/仿真演示、5–600 s 延迟热图、MIKASA-Robo 与 RoboMimic 结果对照、门控激活率可视化及 BibTeX）。
- **通读：** 已完成

## 开源与公开资源核查（2026-11-19）

| 组件 | 状态 | 详情 |
|------|------|------|
| 项目页 | 有 | 展示 RSS 2026 标识、MemMimic / MIKASA-Robo / RoboMimic 实验与门控激活可视化 |
| 核心代码 | **已开源** | [real-stanford/gated-memory-policy](https://github.com/real-stanford/gated-memory-policy)（含 `imitation-learning-policies` 及仿真/真机环境子模块） |
| 数据集与预训练权重 | **已发布** | Hugging Face `yihuai-gao/gated-memory-policy` 提供数据集与 checkpoint |
| 真机部署架构 | **已公开** | 基于 `PolicyServer` 与 `robotmq` 提供低延迟异步推理与历史缓存 |

**结论：** **全链路已开源**（训练、门控标定与训练、仿真环境与真机服务器代码以及数据集和权重均公开）。

## 交叉

- 论文归档：[`sources/papers/gated_memory_policy_arxiv_2604_18933.md`](../papers/gated_memory_policy_arxiv_2604_18933.md)
- 仓库归档：[`sources/repos/real-stanford-gated-memory-policy.md`](../repos/real-stanford-gated-memory-policy.md)
- wiki：[`wiki/entities/paper-gated-memory-policy.md`](../../wiki/entities/paper-gated-memory-policy.md)
