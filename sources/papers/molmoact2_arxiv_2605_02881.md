# MolmoAct2（arXiv:2605.02881）

- **标题：** MolmoAct2: Action Reasoning Models for Real-world Deployment
- **类型：** paper
- **原始链接：** <https://arxiv.org/abs/2605.02881>
- **PDF：** <https://arxiv.org/pdf/2605.02881>
- **项目页/博客：** <https://allenai.org/blog/molmoact2>
- **代码：** <https://github.com/allenai/molmoact2>
- **机构：** Allen Institute for AI（Ai2）、华盛顿大学等
- **作者：** Haoquan Fang、Jiafei Duan（共同一作）等，PI Ranjay Krishna
- **提交：** v1 2026-05-04，v2 2026-05-08（v2 为 51 页含附录；v1 为 32 页无附录）
- **入库日期：** 2026-09-21
- **一句话说明：** 开源 action reasoning VLA 家族：Molmo2-ER 骨干 + flow-matching 连续动作专家；含 base/finetuned 权重、ER 数据集与 LeRobot 集成。
- **核对范围：** 下载 arXiv v1（32 页）与 v2（51 页）两版 PDF，转文本后逐段读完 v1 全文（摘要、§1–§8、参考文献末页）；v2 逐段读了与 v1 不同的部分（§3.4 语言标注、§6.5 OOD 鲁棒性、§6.6–6.8、附录 §A 模型细节、§B.1–B.3 训练细节、§C 评测细节、§D 数据细节、§E 局限）；图 1–14 以图注和图中文字为准，没有逐图看像素；附录 §C 的 RoboEval 逐任务指标大表（Table 17）只核对了几个任务的数字，未逐格核对；§D 的 SO-10x 分布统计表只扫读。没有运行任何代码，也没有读 allenai/molmoact2 仓库。
- **通读：** 已完成

## 开源状态

- **已开源**（Apache-2.0）：`allenai/molmoact2` 训练/评测/部署代码；HF 模型与 MolmoAct2-BimanualYAM 等数据集。论文 §1 与 §8 自述发布权重、训练代码与完整训练数据（本次未逐个检查链接可达性）。

## 核心摘录

1. **Real-world 部署导向：** 论文摘要点出四个痛点——前沿模型闭源、开放权重模型绑定昂贵硬件、带推理的策略延迟过高、微调成功率仍低；MolmoAct2 沿五条轴改进前作 MolmoAct：更强的具身推理 VLM 骨干、新数据集、开放 FAST 动作分词器、重新设计的架构、新的推理范式（§1）。
2. **Molmo2-ER 骨干（§2）：** 从 Molmo2（Qwen3-4B）出发，约 3.3M 样本的具身推理语料（六个能力支柱，Table 1），「先特化再复习」两阶段训练：阶段 1 为 20K 步（含 8% Tulu-3 文本），阶段 2 为 1.5K 步混合训练，p=0.5 取得最佳折中（§2.2）。13 个具身推理基准上 9 个第一，平均 63.8%，比次优 Gemini-ER 1.5 Thinking 高 2.5 点，比 Molmo2 高 17 点（§6.1，Table 3）。
3. **数据（§3）：** MolmoAct2-BimanualYAM 为 34.5k 条演示、超过 720 小时、28 个以上任务，整套采集装置成本低于 6000 美元（图 3）；SO-100/101 数据来自 1,222 个社区 LeRobot 数据集（377 位用户，38,059 条，约 184 小时），经四步过滤（含 TOPReward 质量门）；DROID 过滤后 74,604 条、17,758,044 帧；语言重标注用 Qwen3.5-27B，把唯一指令占比从 22% 提到 46%。
4. **架构与训练（§4）：** 预训练只用离散自回归监督（动作 token 词表 2048，动作补零到 32 维，状态每维 256 档），200K 步、64 张 H100、约 5,760 GPU 小时；后训练加 36 层 DiT 式动作专家，逐层用 VLM 的 KV 缓存做 cross-attention，损失 L_post = L_LM + L_flow，K=4，knowledge insulation，100K 步约 2,300 GPU 小时；微调阶段 K=8、取消 knowledge insulation、仅机器人数据。附录 A.2：发布检查点推理用 N=10 步 Euler。附录 Table 15：图像编码器 380M、连接器 57M、LLM 4.0B、动作专家 621M 参数。
5. **MolmoAct2-Think（§5）：** 深度图用 Depth Anything V2 估计，再经 MolmoAct 的深度 VQ-VAE 量化为 10×10 网格、128 个码值；相邻帧按 32×32 图块余弦相似度（阈值 0.996）判断哪些格子需要重新预测，其余复用缓存。
6. **主要结果（§6）：** 零样本 DROID 仿真 MolmoSpaces 平均 +3.2、MolmoBot +10.6（相对 π0.5-DROID，Table 4/5）；真机 DROID 5 任务 87.1%（对 MolmoBot 48.4、π0.5-DROID 45.2，Table 6）；SO-100 零样本 56.7%（对 π0-SO100/101 45.3，Table 7）；LIBERO 97.2%（Think 98.1%，Table 8）；RoboEval 44.3%；真机 YAM 8 任务平均 50.1%（正文）。消融在 LIBERO 上（v2 的 Table 10–14，v1 为 Table 9–13）。
7. **推理速度（§6.8）：** 单张 H100、LIBERO、horizon 10：MolmoAct2 55.79 Hz，MolmoAct2-Think 12.71 Hz；离散动作路径 14.17 Hz 与 6.82 Hz。
8. **论文自述局限（附录 §E）：** 固定长度动作块开环执行、块边界不保证平滑、55.79 Hz 是摊销吞吐而非闭环反应速度；零样本部署只覆盖 YAM、SO-100/101、DROID 三种本体。

## 与正文不一致或需谨慎之处（核对时发现）

- 真机 YAM 8 任务平均：§6.3 正文写 50.1%，附录 Table 19 的平均列为 50.6（逐任务数值取平均亦为 50.56）；「比次优 OpenVLA-OFT 高 15 点」按 35.5 对 50.6 为 15.1、按 50.1 为 14.6。
- Table 19 的任务名（Cup Stacking、Linearbot、Pegboard、Cup Storing 等）与 §6.3 正文列出的任务名（stack_cups、hang_tools、shelf_cups、make_popcorn 等）并不一一对应；Table 19 的列头写 MolmoAct，而非 MolmoAct2。Table 19 中 Cup Stacking 一项 MolmoAct2 为 54.0，低于 π0.5（62.0）、Cosmos（64.5）与 OpenVLA-OFT（60.5），与正文「8 个任务中 7 个领先」一致。
- 图 6 与 Table 17（RoboEval）的图注和表格行写的是 MolmoAct，附录 §C.1 写的是 MolmoAct2-Post；应是同一模型的命名不统一，论文未说明。
- §6.4 正文「Think 在四个套件中三个提升、一个持平（基线已饱和在 100%）」，Table 8 中 Object 一栏 Think 为 99.8%，MolmoAct2 为 100.0%，即略低 0.2 点而非持平。
- §6 开头问题 4 说要在 LIBERO、MolmoSpace、MolmoBot 上比较 Think 与 MolmoAct2，§6.4 结果只给出 LIBERO。
- Table 9（OOD 鲁棒性）的最后一行标为 MolmoAct2-Think，正文与表题均称 MolmoAct2。
- 摘要说 Think「以远低于既往的延迟保留几何落地」，§6.8 只测了 Think 与 MolmoAct2 自身的速度（12.71 对 55.79 Hz），没有与 MolmoAct 或其他带推理的策略做延迟对比。
- 分词器命名：v1 叫 OpenFAST Tokenizer，v2 叫 MolmoAct2-FAST Tokenizer，同一物。

## 对 wiki 的映射

- [paper-molmoact2](../../wiki/entities/paper-molmoact2.md)
- [molmo-er](../../wiki/entities/molmo-er.md)
- [VLA](../../wiki/methods/vla.md)

## 初次核对记录（2026-11-05，已读 §4 方法、§6.2–§6.8、摘要）

此节为早先的部分核对记录，保留；以本文件顶部「核对范围」为准。

- 标题与作者：arXiv 摘要页标题 *MolmoAct2: Action Reasoning Models for Real-world Deployment*，首作 Haoquan Fang、Jiafei Duan（提交 2026-05-04；PDF 为 v2，2026-05-08）。
- 架构：Molmo2-ER（SigLIP2 ViT + 连接器 + LLM）+ DiT 式动作专家，与 VLM 同为 L=36 层，专家第 l 个 block 对 VLM 第 l 层 K/V（经线性适配器）做 cross-attention（per-layer KV conditioning）；后训练用 knowledge insulation，微调阶段取消。
- 训练：Pre-train 200K 步（FAST 离散动作 token，2048 词表，动作补零 32 维，状态 256 档离散）；Post-train 100K 步，L_post = L_LM + L_flow，K=4 个流采样；微调 K=8。
- 结果（作者表述）：LIBERO 平均 97.2%（Think 变体 98.1%）；RoboEval 44.3%；真机 8 任务平均 50.1%；单张 H100 上 LIBERO horizon 10 的优化后控制频率 55.79 Hz。
- MolmoAct2-Think：只对变化区域重预测深度 token 的自适应深度推理变体；LeRobot 版本不含。
- LeRobot 实现：[lerobot.md](../repos/lerobot.md)。
