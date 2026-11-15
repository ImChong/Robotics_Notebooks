# Predictive Long-Horizon Progress Modeling with Action-Aware Quantification：NimCritic 论文原文归档

> 一手论文阅读记录；本页为结构化摘录，不转载全文。所有实验数字为论文作者报告，未独立复现。

- **标题：** Predictive Long-Horizon Progress Modeling with Action-Aware Quantification / 基于动作感知量化的长时程进度预测建模
- **作者：** Tianchi Xu、Yalin Zheng、Yangxin Zhu、Ruonan Ren、Zheng Fu、Yunlong Wang、Yuanxin Zhong
- **类型：** paper / robot manipulation / process reward modeling
- **论文：** https://arxiv.org/abs/2610.32776
- **HTML 全文：** https://arxiv.org/html/2610.32776v1
- **PDF：** https://arxiv.org/pdf/2610.32776v1
- **版本：** arXiv:2610.32776v1，2026-10-23
- **项目：** https://rimbot.com/research/nimcritic/（[官网访问记录](../sites/nimcritic_rimbot_project.md)）
- **体验：** https://critic.rimbot.com/
- **入库日期：** 2026-11-15
- **核对范围：** 通读 arXiv HTML v1 的 §1–5、公式 (1)–(8)、表 1–5、附录 A（数据组成、训练与推理）、附录 B（评测定义）与附录 C（共享分段、分段消融）；网站与在线 Demo 的可访问性另行核查，不等于源码审阅。
- **通读：** 已完成
- **沉淀到 wiki：** [NimCritic 论文节点](../../wiki/entities/paper-nimcritic.md)

## 摘录 1：进度量化协议（§3.2、公式 1–3）

- **动机：** 单凭阶段关键帧或耗时插值不能一致表示跨任务长时程进度，尤其不能表示尝试、停滞、破坏已完成工作、重新恢复。
- **分解：** 子任务按同一被操作物体的连续动作组织；动作类别权重按全数据集归一化时长估计，动作产生的语义效果决定给分方向。
- **动作记账：** 完成有效进展记全额动作权重，仅实际参与但未完成的尝试记临时半额，其他为零；后续破坏操作按引用的早期动作撤销既有得分，跨子任务也可回撤。
- **局部曲线：** 动作内以时间分数插值；净动作得分 `η_i = α_i w_{c_i} - Σ_{j∈E_i∩I_k} α_j w_{c_j}`；局部进度 `p_t = Σ_i η_i ℓ_i(t) / B`，其中 `B = Σ_i η_i + Σ_{j∈R} w_{c_j}` 包含尚未完成的必要动作。跨子任务保留各原始动作的剩余 credit。
- **标注：** VLM 预标子任务/动作边界、意图与残余动作；人工复核时间边界；规则化后处理算数值。训练数据抽样复核，LOOM 的轨迹全部做时间核验。
- **对 wiki 的映射：** [NimCritic 核心机制](../../wiki/entities/paper-nimcritic.md) 将进度视为可撤销的动作贡献，而非时间流逝比例。

## 摘录 2：预测模型（§3.3–3.4、附录 A.3）

- **长程前端：** 先用 VLM 按物体中心子任务连续分段；同一对象的失败/重试在同段，工具获取与工具操作不同对象拆段，过短片段合并。每段独立评分，按片段时长组合全局进度（公式 4）；全局完成率由原始指令与完整视频单独评分。
- **联合训练：** 冻结视频 VAE、UMT5 文本编码器，双流 diffusion transformer 以 asymmetric mixed attention 关联视频与进度/完成流；Joint Video–Progress Learning（JVPL）通过 flow matching 同时学习未来视频 latent 与进度轨迹。
- **历史扰动：** Historical Progress Perturbation（HPP）向进度历史锚点注入有界扰动，重算相对目标而保持绝对标注不变，减轻递推误差。
- **跨流对齐：** Video–Progress Alignment（VPA）将视频中间特征与真实/去噪进度曲线进行对比对齐；负样本扰动进度而保留完成坐标。
- **工程条件：** 视觉流从 Wan2.2-TI2V-5B 初始化；9 张 2 Hz / 256×448 RGB 帧训练，128,000 updates、48 张 PPU-ZW810E、有效 batch 192。推理为进度流模式：滑窗 5 张已观察 RGB 帧、20 Euler 去噪步、stride=2、重叠加权；作者未提供可直接核验的公开推理源码/权重入口。
- **对 wiki 的映射：** [NimCritic 论文节点](../../wiki/entities/paper-nimcritic.md) 区分训练时未来视频监督与推理时仅使用已观测视频；此处不声称模型已实时部署。

## 摘录 3：规模、评测与基线（§4、表 1–5、附录 A–C）

- **训练规模：** 253,276 episodes，856.95 小时，来自 Bridge、Fractal/RT-1、ARX/DROID/VLABench/LIBERO、EgoDex、SynData、AgiBot 的组合；“800+ h”不是 LOOM 评测集大小。
- **LOOM：** 300 段跨具身轨迹、19,618 人工核准帧；其中单任务 171 段、多任务 129 段。整体 PRC-F1=0.86、TSA=0.92；多任务 PRC-F1=0.95，而单任务仅 0.67，**不优于两种 GVL 单任务 PRC-F1=0.69**。
- **VPB：** 整体 PRC-F1=0.77、TSA=0.82；对照 GVL（GPT-6-Sol）TSA=0.50，相对增幅约 64%（论文近似写 65%），不是 PRC-F1 提升 65%。
- **RoboArena：** 成功 rollout 检索总体 Recall=96.72%、Hit Rate=98.09%，度量是找回人工指定成功执行，不能直接当成机器人任务成功率。
- **消融：** VPB PRC-F1 基线 0.35，加入 JVPL→0.59、HPP→0.67、VPA→0.77；但 TSA 分别为 0.86、0.75、0.87、0.82，并非每个模块使所有指标单调改善。
- **共享分段控制：** LOOM 多任务统一采用 Gemini 3.8 Flash 的缓存子任务分段；NimCritic 0.95 PRC-F1、0.93 TSA；GVL（GPT-6-Sol）PRC-F1=0.96，需避免宣称 NimCritic 在该控制条件下所有指标均第一。
- **分段重要性：** 同一个 critic 不分段的多任务 PRC-F1=0.55，加入分段后 0.95；因此提升有部分来自 VLM 前端与局部评分组合。
- **对 wiki 的映射：** [评测与可比性](../../wiki/entities/paper-nimcritic.md) 明确进度拟合与完成识别是不同任务。

## 摘录 4：边界与应用

- **算法定位：** 该模型是对既有视频 rollout 的 task-progress / completion critic，不能误称为直接输出机械臂动作的 VLA 策略，也不是自带完整的 RL 训练循环。
- **依赖：** 长程视频使用外部 VLM 离线分段，相关评测缓存分段；如果不用该分段前端，跨任务长程性能会不同。
- **部署成本：** 论文列出训练和推理窗口/去噪步数，但未提供可核对的端到端吞吐、实时延迟或边缘硬件成本；不能据此声称可实时闭环。
- **资源开放：** 论文说明方法与数据构成；官网在本次核对中超时、Demo 无可读正文，故不能确认训练数据、LOOM 标注、权重、可运行代码已经发布。
- **对 wiki 的映射：** [工程实践与局限](../../wiki/entities/paper-nimcritic.md) 保留源码与在线 Demo 的核验阻塞。

## 一手参考

- [arXiv v1 HTML 全文](https://arxiv.org/html/2610.32776v1)
- [官方研究页](https://rimbot.com/research/nimcritic/)
- [在线 Demo](https://critic.rimbot.com/)
