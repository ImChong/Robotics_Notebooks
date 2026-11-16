# Rethinking Actions: Continuity-Aware Priors for World-Model Planning / 重新思考动作：面向世界模型规划的连续性感知先验

> 论文一手资料归档（ingest），arXiv:2611.16645v1

- **标题：** Rethinking Actions: Continuity-Aware Priors for World-Model Planning
- **中文标题：** 重新思考动作：面向世界模型规划的连续性感知先验
- **作者：** Shaobo Yang、Zichen Ai、Ziyang Ma、Qian Yang、Bin Fang
- **机构：** 北京邮电大学、阿里巴巴集团（按原文作者栏）
- **类型：** paper
- **论文：** https://arxiv.org/abs/2611.16645
- **HTML 全文：** https://arxiv.org/html/2611.16645v1
- **PDF：** https://arxiv.org/pdf/2611.16645
- **代码：** https://github.com/oliverstack-byte/LG-DKGP
- **版本：** v1，2026-11-12 UTC 提交；本次核对日期 2026-11-14
- **核对范围：** arXiv HTML v1 全文：§1–6（问题、LG-DKGP 方法、主实验、消融、局限）、附录 A–R（补充实验、GP 与实施细节、复现环境、数据来源、训练/规划超参及评测口径）；与官方 README、CLI、训练/规划/评测源码交叉核对。
- **通读：** 已完成
- **开源结论：** 官方 GitHub 已公开训练、准备数据、规划和评测实现；预训练权重、原始训练数据和原生仿真固定场景资产需从 README 指向的外部发布资源分别获取，非克隆仓库即齐备。
- **wiki 映射：** [paper-lg-dkgp.md](../../wiki/entities/paper-lg-dkgp.md)
- **源码归档：** [lg-dkgp.md](../repos/lg-dkgp.md)

## 核心问题与论文贡献

- 专家示范的相邻动作具有统计上的时间依赖。作者在 PushT、Cube、Caddy、Scene2、RoboTwin 五种环境比较真实动作顺序与同 episode 内随机打乱，观察到打乱后相邻标准化动作变化增大。
- 传统视觉世界模型规划可通过搜索动作片段使预测隐状态逼近目标，但无约束动作候选可能丢失专家行为的连续结构。本文目标是**学习专家式动作延续**，不是单纯给动作变化加平滑惩罚。
- LG-DKGP（Locality-Gated Deep-Kernel Gaussian-Process Prior）将**近期和更长历史的已执行动作**通过局部/全局注意力及门控融合，生成神经均值，再由深核 GP 残差修正。
- 使用专家训练数据相邻动作差估计探索尺度，围绕组合均值初始化 CEM；**冻结的 LeWM 世界模型只承担预测与目标评分**。

## 方法核对

1. **输入：** 当前图像和目标图像经冻结视觉编码器得到隐变量；历史为至多 32 条**已执行**动作及有效位掩码，动作按训练集坐标标准化，在线不读取未来专家标签。
2. **门控历史编码器（§3.1）：** 128 宽度、4 头、2 层注意力；全局分支关注所有有效历史，局部分支对较旧动作加年龄惩罚；标量门控自适应融合。历史 token 还包含相邻有效动作增量。
3. **动作先验（§3.2）：** 按未来时间索引生成神经均值；每个动作维度配独立稀疏变分 GP，核由学习到的 query/context RBF 特征和 Matérn-3/2 时间核构成；每坐标 128 个诱导点。
4. **训练（§3.3）：** 先用专家未来动作 MSE 预训练神经均值，再用变分 ELBO 联训编码器、神经均值、GP 残差；视觉世界模型固定不更新。分别以验证 MSE 和联合 NLL 选择检查点。
5. **规划（§3.4）：** 训练动作增量估计各任务的各向同性探索标准差；CEM 每轮 300 候选、30 轮、30 elite，共 9,000 次世界模型候选评分。主实验动作预测长度 H=25、执行长度 P=25、最多 50 步；重新观察、更新已执行动作历史并再次规划。

## 实验数字与评测口径（主文 Table 2）

| 方法 | PushT | Cube | LIBERO Caddy | LIBERO Scene2 | RoboTwin | 五任务 Macro |
|---|---:|---:|---:|---:|---:|---:|
| LeWM | 93.60 | 63.60 | 55.31 | 69.88 | 10.91 | 58.66 |
| iCEM | 96.40 | 69.60 | 54.42 | 72.21 | 4.03 | 59.33 |
| INTACT | 64.40 | 92.00 | 80.97 | **77.53** | 40.26 | 71.03 |
| PRISM（论文原文此名，非同名其他工作） | 94.00 | 87.60 | 56.78 | 56.91 | 13.01 | 61.66 |
| LG-DKGP | **97.60** | **98.80** | **83.32** | 76.96 | **78.66** | **87.07** |

- 表中均为**成功率 %**，五个评测种子 101/202/303/404/505 的均值；Macro 是五任务**不加权**平均。
- **成功定义不全相同：** PushT/Cube 为模拟器原生成功条件；Caddy/Scene2 是演示**片段局部目标**达到；RoboTwin 为局部目标达到或核验过的原生终态成功。不能把 87.07% 解释为五项完整长程任务的统一成功率。
- 论文还报告：去掉 GP 残差使 Macro 下降 3.80 个百分点；固定尺度探索降到 74.92%，专家尺度为 87.07%。Scene2 例外：INTACT 77.53%，LG-DKGP 76.96%。
- RoboTwin 历史长度消融（H=P=25）：L=0 为 2.35%、L=1 为 60.18%、L=5 为 76.92%、L=32 为 81.35%，但短规划时域效果并非单调。
- 目标相对演示位置由 25 步外推到 30–50 步后性能下降；专家动作结构用 transition-feature MMD² 与 variation profile 衡量，但论文指出**常数动作**在部分任务上的 MMD² 可能更低，因此不能把该指标单独当作动作质量结论。

## 局限与复现边界

- 结果来自五个**模拟**任务；论文没有给出本文策略的真机落地成功率或低延迟实时性承诺。
- 视觉条件、历史长度和规划时域的收益依任务而变；局部目标定义限制了与完整长程成功率的直接比较。
- 主设置每次规划 9,000 次候选评分，属于较高搜索预算；不能按无搜索策略延迟理解。
- 复现需 Linux x86-64 / Python 3.10 / 单 CUDA GPU，按任务分别安装 pc、libero 或 robotwin 依赖；配对的 LeWM 与先验权重不得跨任务混用。大型数据与模型权重另行下载。当前归档是**原文与源码静态核对**，未实际训练或复现表格数据。

## 对 wiki 的映射

- [LG-DKGP 论文独立节点](../../wiki/entities/paper-lg-dkgp.md)：连续性感知先验与冻结世界模型的职责分离、两阶段训练、CEM 在线时序、评测指标及边界。
- [开源仓库归档](../repos/lg-dkgp.md)：可执行入口、文件路径与复现限制。
- 关联已有 [LeWM](../../wiki/entities/paper-lewm.md)、[INTACT](../../wiki/entities/paper-intact.md)、[MPC](../../wiki/methods/model-predictive-control.md)。

## 一手资料

- [arXiv HTML v1（含实验附录）](https://arxiv.org/html/2611.16645v1)
- [官方代码 README](https://github.com/oliverstack-byte/LG-DKGP/blob/main/README.md)
- [官方复现数据说明](https://github.com/oliverstack-byte/LG-DKGP/blob/main/docs/data.md)
