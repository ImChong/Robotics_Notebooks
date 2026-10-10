# MotionSpaceFlow 官方项目页（MSFlow）

> 来源归档（最近复核：2026-10-10）

- **类型：** project page / research demo
- **项目页：** <https://yu1ut.com/MSFlow-HP/>
- **论文：** [MotionSpaceFlow（arXiv:2609.34190）](https://arxiv.org/abs/2609.34190)
- **代码：** [lycorp-jp/MSFlow](https://github.com/lycorp-jp/MSFlow) — 代码仓 README 将其标记为项目的临时开源实现
- **权重：** [ly-corporation/MSFlow（Hugging Face）](https://huggingface.co/ly-corporation/MSFlow)
- **作者与机构：** Qing Yu、Kent Fujiwara；LY Corporation
- **arXiv 日期：** 2026-09-28
- **一句话说明：** MotionSpaceFlow 在原始连续动作表示中进行文本条件 flow matching；全局 XYZ 变体支持推理阶段对任意关节和帧施加空间约束。

## 项目页和论文要点

- 项目主题是直接在动作空间生成全身运动，不使用学习式动作 latent 编码器/解码器。
- 展示文本条件动作生成和推理期关节/帧约束；官方论文展示如局部肢体定位、向前行走、绕圈和侧手翻等控制样例。
- 263D 增量表示和全局 XYZ 表示采用不同的时序注意力：前者 causal，后者 bidirectional。
- 结果按 HumanML3D、SnapMoGen 各自评测协议报告；单个指标不能代替整体对比。完整数字和适用边界见[论文归档](../papers/motionspaceflow_arxiv_2609_34190.md)。

## 开源核查（2026-10-10）

- 项目页链接到官方 [GitHub](https://github.com/lycorp-jp/MSFlow) 与 [Hugging Face 权重](https://huggingface.co/ly-corporation/MSFlow)。
- 代码仓提供安装、预训练模型下载、HumanML3D/SnapMoGen 数据准备、demo、训练与评测命令；主 README 标注仓库为**临时开放**，未来可能变只读或私有。
- 代码许可为 CC0 1.0；仓库另含第三方软件，须遵循 [NOTICE.txt](https://github.com/lycorp-jp/MSFlow/blob/main/NOTICE.txt) 及对应上游条款。
- 当前开放代码是人体动作生成的研究实现，不是机器人控制或物理可行性保证。

## 对 wiki 的映射

- [MotionSpaceFlow 项目实体](../../wiki/entities/paper-motionspaceflow.md)
- [官方代码仓归档](../repos/msflow-lycorp-jp.md)
- [论文来源归档](../papers/motionspaceflow_arxiv_2609_34190.md)
