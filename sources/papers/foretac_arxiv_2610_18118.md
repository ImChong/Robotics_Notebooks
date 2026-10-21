# ForeTac: Predictive Contact Guidance for Generative Robot Policies（arXiv:2610.18118）

> 来源归档（ingest）

- **标题：** ForeTac: Predictive Contact Guidance for Generative Robot Policies
- **arXiv：** <https://arxiv.org/abs/2610.18118>（PDF：<https://arxiv.org/pdf/2610.18118>；提交于 2026-10-13，v1；HTML 版 404，数字取自 PDF 文本）
- **代码：** 无（项目页标注 "Coming soon"，见开源状态）
- **项目页：** <https://foretac.github.io/> — 归档见 [`sources/sites/foretac-project.md`](../sites/foretac-project.md)
- **作者：** Zhiyuan Chen, Shuai Chen, Haoran Zhou, Yizhe Zhao, Kaiyuan Yang, Zhixiang Duan, Shaoqing Xu, Ran Cao, Diyun Xiang, Yangwei You
- **机构：** PDF 首页未列单位，项目页为匿名（"Anonymous Institution"），**未核实**
- **入库日期：** 2026-10-17
- **一句话说明：** 在冻结的扩散 / flow matching 策略上，预测候选动作的未来触觉，用接触质量打分并把梯度回传到采样状态来引导动作；四个真机任务平均成功率 36.3%→71.3%。

## 开源状态（步骤 2.5）

- **项目页核查（2026-10-17）：** Paper 与 Code 均标 "Coming soon"，无 arXiv、GitHub 链接；概览视频 "Video unavailable"；作者与单位匿名。
- **结论：** **宣称将开放 / 未列链接**：截至入库日项目页未列 GitHub。论文 PDF 未给代码地址（以项目页为准）。

## 摘录 1：方法

- 表示-预测-评估-引导环：TacVAE（8 帧 9×9×2 marker 位移历史 → 16×3×3 潜在图，144 维）；Foresight Transformer（3 层，输入当前触觉 token、本体状态与候选动作块，一次前向预测 H=16 步未来触觉潜变量）；接触质量打分器（每任务单独训练，4 个接触类：1 个目标接触 + 3 个录得失败模式；优化目标是好-坏 logit margin）。
- 引导：把策略的临时干净动作估计送入 Foresight 与打分器，梯度回传到当前扩散 / flow 状态，归一化后更新；基座、TacVAE、Foresight、打分器在部署时全部冻结。
- 超参：扩散骨干 30 采样步，在 10 步上施加引导，尺度 0.003，每次执行生成块的前 8 个动作后重规划；π0.5 在 10 个积分步的最后 2 步引导。
- 预训练语料：5,969 条时间戳对齐 episode，约 210 万 marker 帧。

**对 wiki 的映射：** 核心原理、工程实践。

## 摘录 2：实验

- 平台：Realman RM75 + Robotiq Hand-E，Vitai GF225 光学视触觉传感器，两台 RealSense D405；四任务（平板擦拭、曲面花瓶擦拭、刷卡、易碎芯片抓取），每方法每任务 20 次，四任务平均（共 80 次）。
- Table I（SR/CSR 平均）：DP 36.3/20.0；RDT 50.0/40.0；DP+Tactile Concat 48.8/41.3；RDP 58.8/48.8；DP+ForeTac 71.3/66.3；π0.5 46.3/36.3；π0.5+ForeTac 67.5/62.5。
- Table II 消融（平均 SR/CSR）：ResNet 输入 50.0/38.8；仅预测辅助损失 55.0/48.8；预测重排序 58.8/52.5；当前接触引导 60.0/51.3。
- 偏好序准确率：记录触觉 91.2%，预测触觉 92.9%（5,640 对）；引导诊断（项目页）293/305 次配对更新提升 margin（96.1%）。
- 延迟：RTX 4090 上单次重规划 DP 132.46 ms，ForeTac 261.26 ms（项目页同数字）；基座 338.0M 参数，ForeTac 增加 34.9M（项目页）。
- 有效预测视野 L：L=1 时 Board 任务 SR/CSR 60/50，L=16 时 85/80。

**对 wiki 的映射：** 实验与评测、结论。

## 摘录 3：作者自述局限

- 接触质量模型为每任务单独训练（目标接触 + 录得失败模式）；重复梯度计算增加重规划延迟。
- 未来方向：可迁移质量模型、不确定性感知触觉预测、自动难负例采集、更高效引导。

## 建议 wiki 动作

- 新建 `wiki/entities/paper-foretac.md`；仅新增实体页。
