# MDM: Human Motion Diffusion Model

- **类型：** 作者一手论文
- **作者：** Guy Tevet, Sigal Raab, Brian Gordon, Yonatan Shafir, Daniel Cohen-Or, Amit H. Bermano
- **会议：** ICLR 2023
- **机构：** 特拉维夫大学（Tel Aviv University）
- **论文：** https://arxiv.org/abs/2209.14916
- **原始版本：** arXiv:2209.14916（以所列论文链接版本为准）
- **项目页：** https://guytevet.github.io/mdm-page/
- **代码：** https://github.com/GuyTevet/motion-diffusion-model
- **核对范围：** arXiv:2209.14916v2 全文：§1 引言、§2 相关工作、§3 x0 预测与条件扩散/几何损失/引导/时空 inpainting、§4 HumanML3D/KIT/HumanAct12/UESTC 实验表1–4、§5 无条件合成与编辑表5、§6 局限和约一分钟推理；参考文献表浏览。该 HTML 版无独立附录。
- **通读：** 已完成

## 方法与应用

采用 classifier-free 条件生成与 Transformer 去噪；预测原始动作样本而非直接噪声，使关节位置、速度和足接触相关损失可参与训练。它主要生成给定时长的动作序列。

## 实验与结论

原文在 HumanML3D、KIT-ML 等文本动作基准与动作类别条件任务上评测；官方仓库特别提示早期评价代码曾修复，不应跨版本直接抄旧表。

## 局限 / 对 wiki 的映射

文生运动学序列并不包含真实机器人的接触估计和控制反馈；原始 MDM 的单次整段生成不等价于 0.5 秒视觉闭环。

- [MDM：文本与动作条件人体运动扩散](../../wiki/entities/paper-mdm-human-motion-diffusion.md)

## 原文复核摘录（正文证据）
- **关键机制：** 不是逐帧自回归。MDM 在给定动作序列长度上对**整段动作**做扩散，Transformer 预测 clean motion x0，CFG 让文本或动作类条件与无条件采样共存。
- **损失边界：** 位置、速度和足接触几何项在有 joint rotations/FK 的场景中使用；论文 §4.1 说明 HumanML3D/KIT 的显式表示**未再施加几何损失**，不可笼统说所有基准都用该项。
- **结果：** HumanML3D FID=0.544；KIT FID=0.497（以 arXiv v2 表1/2 为准），早期论文一次约 1000 次前向、生成一段约 1 分钟，实时交互不在其评测结论中。
- **参考：** §3、§4.1–§4.2、§5.1–§5.2、§6；[论文原文](https://arxiv.org/html/2209.14916)。
