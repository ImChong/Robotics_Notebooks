# Looped World Models（循环世界模型）

> 一手论文阅读归档（区别于 `sun_awesome_wm_...` 策展索引）

- **类型：** paper / technical report
- **标题：** Looped World Models（循环世界模型）
- **机构：** FaceMind Research Asia（PDF 首页署名）
- **作者：** Hongyuan Adam Lu、Z.L. Victor Wei 等 31 位作者（完整名单见 arXiv）
- **发布日期：** 2026-06-16
- **版本：** arXiv:2606.18208v1
- **论文：** <https://arxiv.org/abs/2606.18208>
- **PDF：** <https://arxiv.org/pdf/2606.18208>
- **HTML：** <https://arxiv.org/html/2606.18208v1>
- **核对范围：** PDF v1：§1–3 架构、训练、早退和延迟解码；§4–6 实验、讨论和公开边界；正文与后续表格（ScienceWorld、ALFWorld、Step 1–5），参考文献；核对 Fig. 1、Tables 1–5 的指标口径。
- **通读：** 已完成
- **官方源码：** arXiv 摘要页和论文正文未给出可核实的官方训练/推理仓库入口；截至 2026-11-16 未确认官方代码、权重或项目主页。第三方同名复现仓不视为作者源码。
- **沉淀到 wiki：** [`wiki/entities/paper-sa-2606-18208-looped-world-models.md`](../../wiki/entities/paper-sa-2606-18208-looped-world-models.md)

## 原文机制摘录（归纳，非逐字转存）

1. **动作条件潜状态预测**（§3.1）：观测编码器 `E_phi(o_k)`、动作编码器 `A_psi(a_k)` 和上一环境步的隐藏状态 `h_(k-1)` 进入动力学核心；预测头解码下一观测（或潜目标）、奖励及 continuation。
2. **双层循环**（§3.2）：Prelude 产生条件信号；共享 Transformer 的 Recurrent Block 在**同一个环境时间步内**重复迭代 T 次细化 latent；Coda 输出供下一环境步传播。**内循环不是物理时间步**。
3. **谱约束**（Eq. 5–7）：保留矩阵由负对角连续动力学离散化，`Abar = exp(Delta * diag(-exp(a)))`，因对角元素在 `(0,1)`，保证**线性保留项**收缩；不可直接推导现实系统一定稳定。
4. **可变计算**（§3.3–3.4）：训练时每序列采样 Poisson 迭代深度；推理用 sigmoid gate 和阈值执行早退。对简单/复杂转移按需分配循环次数。
5. **延迟解码**（§3.5）：K 步候选动作依次注入潜空间，期间不反复调用完整解码头；最后仅解码 `h_K`。训练使用末端监督、潜一致性和状态步幅约束；K 从 1 递增的课程学习缓解梯度不稳。
6. **规划接口**（§3.5.5）：CEM / MPPI 等规划器可提供候选动作序列，LoopWM 在潜空间推演并评价末端状态；这不等同于论文已做真实机器人闭环部署。

## 实验事实与边界

| 基准 | LoopWM（约 1B 参数） | 对照 | 评价口径 |
|---|---|---|---|
| ScienceWorld，Table 2，连续 5 动作后 | EM 68.4%，F1 85.3%，BLEU-4 80.7%，Entity 83.9% | Claude Opus 4.6 Max：EM 47.2%，F1 72.8%，BLEU-4 64.4%，Entity 72.3% | 生成/预测文本状态的匹配指标，不是真机成功率 |
| ALFWorld，Table 4，连续 5 动作后 | EM 51.6%，F1 80.4%，BLEU-4 71.6%，Entity 81.1% | Claude Opus 4.6 Max：EM 53.0%，F1 72.6%，BLEU-4 66.8%，Entity 77.0% | EM 略低于该对照，不应表述成全部指标领先 |

- 摘要中的 **up to 100× parameter efficiency** 是作者的结论，不等于实测的 100 倍推理加速、训练数据减少 100 倍或真实机器人交互成本下降 100 倍。
- 表 5 与后续按任务/步数的表格报告延迟解码相对特定 LLM 基线的提升；并非明确的「LoopWM 无延迟解码」对照消融，不可据此单独归因。
- §6 作者说明更广泛的连续视觉环境结果**未在此版本完整公开**；缺少可查的真实机器人控制、接触动力学、闭环规划成功率/采样效率对比，以及官方复现入口。
- 谱半径 < 1 的形式化约束作用于受控线性部分，非对任意非线性网络或任意真实环境轨迹的无条件保障。

## 与既有策展归档的关系

- [Awesome World Models 第 118 条索引](sun_awesome_wm_2606_18208_looped-world-models.md) 只作外部清单坐标；本页为原始论文阅读归档。
- [实体详情节点](../../wiki/entities/paper-sa-2606-18208-looped-world-models.md) 汇总定义、机制图、实验、适用边界与延伸阅读。
