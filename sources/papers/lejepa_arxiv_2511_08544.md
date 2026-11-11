# LeJEPA: Provable and Scalable Self-Supervised Learning Without the Heuristics（arXiv:2511.08544）

> 来源归档（ingest）

- **标题：** LeJEPA: Provable and Scalable Self-Supervised Learning Without the Heuristics
- **类型：** paper / JEPA / self-supervised / SIGReg
- **arXiv：** <https://arxiv.org/abs/2511.08544>
- **HTML 全文：** <https://arxiv.org/html/2511.08544v3>
- **本次核对版本：** arXiv v3（2025-11-14；与 v1 提交日 2025-11-11 区分）
- **代码：** <https://github.com/rbalestr-lab/lejepa>（CC BY-NC 4.0）
- **作者：** Randall Balestriero、Yann LeCun
- **机构：** 布朗大学；纽约大学；Meta-FAIR
- **入库日期：** 2026-09-05
- **复核日期：** 2026-11-11
- **核对范围：** arXiv v3 全文第 1–7 节（背景、线性与非线性 probe 的理论条件、SIGReg、LeJEPA 目标、实验及结论）；附录 A–F（kNN/核探针、B.1–B.14 证明、统计学背景与补充表格/图形）；交叉核对官方 MINIMAL.md、README 与 LICENSE。
- **通读：** 已完成
- **一句话说明：** 证明各向同性高斯是 JEPA embedding 的最优下游风险分布，并用 SIGReg（随机投影 + 特征函数检验）把坍塌从启发式换成可证明约束。

## 开源状态（2026-09-05）

- **已开源：** [rbalestr-lab/lejepa](https://github.com/rbalestr-lab/lejepa)（1337★）。`lejepa` 包 + ImageNet 最小例（`MINIMAL.md`）。
- **许可证：** **CC BY-NC 4.0**（不可商用）。
- 项目页：仓库即入口；无独立 `.github.io`。

## 核心论文摘录（MVP）

### 1) 各向同性高斯是探针最优

- **链接：** §3
- **摘录要点：** 线性 / 非线性探针下，各向同性高斯唯一最小化下游预测风险；各向异性同时放大偏差与方差。
- **对 wiki 的映射：** [paper-lejepa](../../wiki/entities/paper-lejepa.md)

### 2) SIGReg 线性复杂度

- **链接：** §4
- **摘录要点：** 随机 1D 投影 + Epps–Pulley 等特征函数检验；时间/内存对维数与 batch 线性。
- **对 wiki 的映射：** [paper-lewm](../../wiki/entities/paper-lewm.md) 把 SIGReg 接到动作条件 WM。

### 3) 启发式卸掉

- **链接：** §5–6
- **摘录要点：** 无 stop-gradient / teacher–student / EMA；单超参；ViT-H/14 IN1K 冻结线性探针 **79%**；Galaxy10 域内预训练超过 DINOv2/v3 迁移。
- **对 wiki 的映射：** [paper-levjepa](../../wiki/entities/paper-levjepa.md)

## 本次全文复核补充（2026-11-11）

- **理论条件：** 在固定协方差规模/总方差等约束和相应回归/平滑性假设下，讨论任意未知下游目标的风险；线性 probe 的协方差各向同性是关键，非线性 kNN 与核 probe 的结论需依赖附录 A/B 的额外正则性条件。勿误读成对任意任务、任意训练网络均无条件最优。
- **SIGReg 的识别性与可实现性：** 依靠 Cramér–Wold 方向投影和一维特征函数的 Epps–Pulley 检验；与高阶矩及 CDF 排序方法相比，作者强调梯度有界、DDP 易并行和按固定投影数/积分节点数近似线性扩展。有限随机方向、数值积分和有限 batch 仍是近似；第 5 节注明 minibatch 带来随 `O(1/N)` 缩小的统计偏差。
- **模型目标：** 第 5 节式 (9) 为 `(1-λ) × global-view prediction loss + λ × SIGReg`，所有视图对齐全局视图均值；主方法不需要独立 predictor、EMA teacher 或 stop-gradient。论文在 ViT 上允许用 SWA 作为可选的性能增强，但不是防坍塌必要条件。
- **有出处的起始超参数：** 第 6.1 节推荐 `λ=0.05`、2 张 global + 8 张 local views、batch ≥128；SIGReg 推荐 1024 随机方向、17 个积分节点与 `[-5,5]` 区间。这些是论文实验建议，不是所有数据或架构的精确最优值。官方 README 的代码示例可能采用 2+6 视图，勿与论文推荐混为同一固定值。
- **评测范围：** 冻结线性 probe、少样本/全量微调、In-domain Galaxy10/Food101、训练 loss 与下游相关性、不同模型与规模稳定性；原文不包含机器人实体闭环控制或动作条件系统的成功率测量。
- **代码地址更新：** 原 `rbalestr-lab/lejepa` 地址现重定向到 <https://github.com/galilai-group/lejepa>；官方 LICENSE 核对为 CC BY-NC 4.0，使用时仍需遵循非商业限制。
