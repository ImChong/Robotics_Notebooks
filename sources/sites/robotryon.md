# RoboTryOn 官方项目主页

> 来源归档（site），与 [paper-robotryon](../../wiki/entities/paper-robotryon.md) 复用论文唯一身份。

- **标题：** RoboTryOn: Synthesizing Dexterous Robot Demonstrations from Human Videos via Virtual Try-On
- **类型：** site
- **项目主页：** <https://ryan-seungyong-lee.github.io/RoboTryOn/>
- **论文：** <https://arxiv.org/abs/2611.05163>
- **代码：** <https://github.com/ryan-seungyong-lee/RoboTryOn>
- **权重：** <https://huggingface.co/RyanL22/RoboTryOn>
- **交互示范：** <https://huggingface.co/spaces/RyanL22/RoboTryOn>
- **入库日期：** 2026-11-05
- **复核日期：** 2026-11-22
- **核对范围：** 官方网页头部链接、Abstract、Method、One Human Video Various Embodiments、Learning New Tasks from Human Video、Analysis（视觉与动作差距）、BibTeX。
- **通读：** 已完成
- **沉淀到 wiki：** [唯一详情](../../wiki/entities/paper-robotryon.md)

## 官方项目页核对

- **开源状态：** 截至 2026-11-22，Code 指向公开 GitHub 仓库，Checkpoints 指向公开 Hugging Face，另有 Demo；旧入库记录的“两个 Coming soon 按钮”已不符合本次核对状态。
- **机制：** 关键帧人手→机器人修补；人手骨架流引导视频补间；深度特征 IDM 为合成机器人视频预测关节动作，辅以腕部校正。
- **评测：** OpenArm、RB-Y1 的七个没有对应机器人遥操作演示的任务，加入合成视频的数据后，平均表现约由 2% 到 60%；作者展示 OpenArm + RH56F1、RB-Y1 + WujiHand2、GR-1 的视频生成样例，但后者未报告对应真机策略成功率。
- **复现边界：** 详细数据与局限须以 [原论文](../papers/robotryon_arxiv_2611_05163.md) 为准；官方仓库指引见 [源码归档](../repos/robotryon.md)。
