# Igniting VLMs toward the Embodied Space（WALL-OSS，arXiv:2509.11766）

> 来源归档（paper）

- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2509.11766>（v1 提交 2025-09-15，论文页写 Date: September 8, 2025）
- **PDF：** <https://arxiv.org/pdf/2509.11766>
- **作者：** Andy Zhai, Brae Liu, Bruno Fang, Chalse Cai, Ellie Ma 等（X Square Robot；Lucy Liang 项目负责人，Hao Wang 通讯）
- **代码：** <https://github.com/X-Square-Robot/wall-x>；项目页 <https://x2robot.com/en/research/68bc2cde8497d7f238dde690>
- **入库日期：** 2026-11-05
- **再核日期：** 2026-11-21
- **线索来源：** LeRobot 固定提交 `9e5ee98` 的 `docs/source/walloss.mdx`（`policy.type=wall_x`）
- **核对范围：** 通读 `https://arxiv.org/html/2509.11766`（v1）全文，含摘要、§1 引言、§2 相关工作、§3 方法（MoE 静态路由、Inspiration / Integration 两阶段、Uni-CoT）、§4 数据构成（Table 1）、§5 评测与消融（Table 2–3、Fig. 6–7）及 §6 讨论；未运行真机实验
- **通读：** 已完成

## 核心摘录（已通读全篇正文 §1–§6 与 Table 1–3）

- 针对 VLM 迁到具身空间的三处鸿沟（模态与数据、预训练分布、训练目标），提出紧耦合 MoE（静态路由，按 token 类型分流 VL FFN / Action FFN）。
- 两阶段：Inspiration（复用原 FFN，具身 VQA + FAST 离散动作 token 的交叉熵）→ Integration（换成 flow matching 的连续动作头，先冻结 VLM 再联合训练）。
- Unified Cross-Level CoT：指令推理、子目标分解、动作合成在同一可微框架内。
- 数据：论文写语料「超过 10,000 小时」。
- 论文本身没有给出 LeRobot 集成的细节；LeRobot 文档称其实现改编自 wall-x 仓库。

## 对 wiki 的映射

- [cn-os-wall-x](../../wiki/entities/cn-os-wall-x.md)（WALL-OSS 一代的 canonical 节点；二代见 [paper-wall-oss-0-5](../../wiki/entities/paper-wall-oss-0-5.md)）
- LeRobot 实现：[lerobot.md](../repos/lerobot.md)
