# MolmoAct2（arXiv:2605.02881）

- **标题：** MolmoAct2: Action Reasoning Models for Real-world Deployment
- **类型：** paper
- **原始链接：** <https://arxiv.org/abs/2605.02881>
- **项目页/博客：** <https://allenai.org/blog/molmoact2>
- **代码：** <https://github.com/allenai/molmoact2>
- **机构：** Allen Institute for AI（Ai2）
- **入库日期：** 2026-09-21
- **一句话说明：** 开源 action reasoning VLA 家族：Molmo2-ER 骨干 + flow-matching 连续动作专家；含 base/finetuned 权重、ER 数据集与 LeRobot 集成。

## 开源状态

- **已开源**（Apache-2.0）：`allenai/molmoact2` 训练/评测/部署代码；HF 模型与 MolmoAct2-BimanualYAM 等数据集。

## 核心摘录

1. **Real-world 部署导向：** 相对 frontier 闭源 VLA，强调可复现 fine-tune、真机 Franka/SO-100/YAM 与 ManiSkill 零样本评测。
2. **Molmo2-ER 骨干：** 具身推理 VLM 与动作头解耦；ER 数据集单独发布。
3. **生态：** HuggingFace LeRobot 官方 MolmoAct2 policy；MolmoSpace leaderboard 第一 VLA。

## 对 wiki 的映射

- [paper-molmoact2](../../wiki/entities/paper-molmoact2.md)
- [molmo-er](../../wiki/entities/molmo-er.md)
- [VLA](../../wiki/methods/vla.md)

## 论文正文核对（2026-11-05，已读 §4 方法、§6.2–§6.8、摘要）

- 标题与作者：arXiv 摘要页标题 *MolmoAct2: Action Reasoning Models for Real-world Deployment*，首作 Haoquan Fang、Jiafei Duan（提交 2026-05-04；PDF 为 v2，2026-05-08）。
- 架构：Molmo2-ER（SigLIP2 ViT + 连接器 + LLM）+ DiT 式动作专家，与 VLM 同为 L=36 层，专家第 l 个 block 对 VLM 第 l 层 K/V（经线性适配器）做 cross-attention（per-layer KV conditioning）；后训练用 knowledge insulation，微调阶段取消。
- 训练：Pre-train 200K 步（FAST 离散动作 token，2048 词表，动作补零 32 维，状态 256 档离散）；Post-train 100K 步，L_post = L_LM + L_flow，K=4 个流采样；微调 K=8。
- 结果（作者表述）：LIBERO 平均 97.2%（Think 变体 98.1%）；RoboEval 44.3%；真机 8 任务平均 50.1%；单张 H100 上 LIBERO horizon 10 的优化后控制频率 55.79 Hz。
- MolmoAct2-Think：只对变化区域重预测深度 token 的自适应深度推理变体；LeRobot 版本不含。
- LeRobot 实现：[lerobot.md](../repos/lerobot.md)。
