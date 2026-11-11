# MDM: Human Motion Diffusion Model

- **类型：** 作者一手论文
- **作者：** Guy Tevet, Sigal Raab, Brian Gordon, Yonatan Shafir, Daniel Cohen-Or, Amit H. Bermano
- **会议：** ICLR 2023
- **机构：** 特拉维夫大学（Tel Aviv University）
- **论文：** https://arxiv.org/abs/2209.14916
- **原始版本：** arXiv:2209.14916（以所列论文链接版本为准）
- **项目页：** https://guytevet.github.io/mdm-page/
- **代码：** https://github.com/GuyTevet/motion-diffusion-model
- **核对范围：** 核对论文官方页面/HTML 可见的方法与结果、作者项目主页；并核对仓库 README 与部分运行入口；全文细节、所有附录未逐段完成交叉核对。
- **通读：** 未完成

## 方法与应用

采用 classifier-free 条件生成与 Transformer 去噪；预测原始动作样本而非直接噪声，使关节位置、速度和足接触相关损失可参与训练。它主要生成给定时长的动作序列。

## 实验与结论

原文在 HumanML3D、KIT-ML 等文本动作基准与动作类别条件任务上评测；官方仓库特别提示早期评价代码曾修复，不应跨版本直接抄旧表。

## 局限 / 对 wiki 的映射

文生运动学序列并不包含真实机器人的接触估计和控制反馈；原始 MDM 的单次整段生成不等价于 0.5 秒视觉闭环。

- [MDM：文本与动作条件人体运动扩散](../../wiki/entities/paper-mdm-human-motion-diffusion.md)
