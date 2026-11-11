# Parallel-Attention Based Myoelectric Gesture Recognition Network for Daily Manipulation of Bionic Prosthetic Hands（JBE 2026）

> 来源归档（paper / 待补全文）：基于 Springer Nature 官方论文页的摘要、元数据及数据可用性声明整理；**并非全文精读**，没有重新分发受版权保护的论文。

- **论文标题：** Parallel-Attention Based Myoelectric Gesture Recognition Network for Daily Manipulation of Bionic Prosthetic Hands
- **中文标题：** 基于并行注意力的肌电手势识别网络：面向仿生假手日常操作
- **类型：** paper / sEMG / bionic prosthetic hands / gesture recognition
- **作者：** Boao Li、Ziming Chen、Shixian Wang、Fuchun Sun、Bin Fang、Du Jiang、Bo Tao、Gongfa Li
- **期刊：** Journal of Bionic Engineering（2026）
- **发表日期：** 2026-10-20（Springer 官方标注的在线发表日）
- **DOI：** 10.1007/s42235-026-01012-7
- **原文：** <https://link.springer.com/article/10.1007/s42235-026-01012-7>
- **DOI 入口：** <https://doi.org/10.1007/s42235-026-01012-7>
- **出版商 PDF 入口：** <https://link.springer.com/content/pdf/10.1007/s42235-026-01012-7.pdf>（当前无法公开获取全文，非免费全文链接）
- **入库日期：** 2026-11-11
- **通读：** 未完成
- **核对范围：** 已核对 Springer 官方摘要、作者名单、在线发表日期、数据与材料可用性声明。论文正文方法、消融、实验协议、完整图表及附录受订阅访问限制，未获取和通读；因此暂不绘制声称来自全文的 Mermaid 主流程图。
- **一句话说明：** 针对日常操作时手臂姿态改变使肌电手势识别精度下降的问题，论文提出并行双分支注意力 DCNN-GRU 模型，并在 8 位健全参与者与 2 位截肢参与者的数据上进行评估。

## 已核实的核心内容（仅依据出版商摘要）

1. **任务动机：** 日常仿生假手操作伴随动态手臂动作，手臂位置变化使表面肌电（sEMG）特征分布变化，影响手势分类的准确性和鲁棒性。
2. **模型结构：** 作者称其为 **DCNN-GRU**（Dynamic Convolutional Neural Network–Gated Recurrent Unit）；两条并行分支对同一窗口级特征图分别学习 **全局通道权重**和**位置分辨的通道—空间门控**，GRU 处理窗口表征的有序序列。
3. **受试者：** 8 位健全参与者 + 2 位截肢参与者；摘要未披露所有细分实验协议。
4. **离线报告：** 在六种条件全部参与训练的设置（ALL）下，汇总测试窗口准确率为 **96.76%**；这是窗口级指标，不等于连续实际操作的整任务成功率。
5. **在线报告：** 受试者特定的三项日常活动评估，识别准确率均**超过 86%**；不能与离线 96.76% 直接当作同一指标对比。

## 开放性与证据边界（核查日期：2026-11-11）

| 项 | 已核对状态 |
|----|------------|
| 正式论文页 | [Springer 官方](https://link.springer.com/article/10.1007/s42235-026-01012-7)，摘要公开，正文需订阅 |
| 免费完整原文 | 本次未能取得；未找到可核验的作者开放版本 |
| 项目主页 | 本次未找到与该论文独立对应的官方项目页 |
| 训练 / 推理源码 | 本次未找到作者官方可运行仓库；**不能据此断言从未发布** |
| 模型权重 | 未核实有公开发布 |
| 数据与材料 | Springer 声明：可向通讯作者提出合理请求，**并非公开下载** |
| Mermaid 方法流程 | **待补**：未完成全文通读，不能据摘要填补未知模块边界 |
| 源码运行时序图 | **不适用（现阶段）**：尚无已核实的官方可运行代码及入口 |

## 完成全文 ingest 前待核项目

- 获取合法的全文访问途径，逐段阅读网络张量形状、注意力融合、窗口/时序结构和训练目标，并核对方法图与公式。
- 核对六种条件的具体含义、受试者划分、在线三项任务、基线、混淆矩阵、消融和复杂度指标。
- 查验作者是否另行公开数据/代码；如有运行入口，先读源码，再绘制源码时序图。
- 在核实后补充 Mermaid 原文主流程图，把 wiki 状态从 draft 改为 complete，并重新跑节点健康度。

## 对 wiki 的映射

- [并行注意力 DCNN-GRU 肌电手势识别](../../wiki/entities/paper-parallel-attention-myoelectric-gesture-recognition.md)
- [DexEMG：肌电连续手部姿态遥操作（相关方向，不是相同任务）](../../wiki/entities/paper-dexemg.md)

## 参考来源

- [Springer Nature 原始论文记录（摘要、元数据、数据声明）](https://link.springer.com/article/10.1007/s42235-026-01012-7)
- [DOI](https://doi.org/10.1007/s42235-026-01012-7)
