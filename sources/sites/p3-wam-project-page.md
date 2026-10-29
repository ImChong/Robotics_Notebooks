# P³-WAM 官方项目页

> 来源归档（site / project-page）

- **标题：** Perceive, Predict, and Plan: A Unified World Action Model for Egocentric Dexterous Manipulation
- **类型：** site / project page
- **官方入口：** <https://p3-wam.pages.dev/>
- **论文：** <https://arxiv.org/abs/2610.22363>
- **论文 PDF：** <https://p3-wam.pages.dev/paper.pdf>
- **主文 PDF：** <https://p3-wam.pages.dev/paper-main.pdf>
- **入库日期：** 2026-10-27
- **一句话说明：** P³-WAM 在统一模型中完成第一视角手部运动重建、动作条件视频预测，以及由视觉目标引导的视频与动作联合规划。

## 页面公开信息

项目页将模型拆成三种输入/输出模式：

| 模式 | 输入 | 输出 |
|------|------|------|
| Perceive | 第一视角视频 | 3D 手部运动与相机轨迹 |
| Predict | 初始图像、指令、指定手部/相机动作 | 后续视频 |
| Plan | 初始图像、指令、可选目标图像 | 未来视频与 3D 动作轨迹 |

统一训练时，视频与动作只部分可见，模型预测缺失帧或动作状态；训练采用随机可见性掩码。论文介绍在预训练视频扩散 Transformer 上使用混合 Transformer 结构，共享基础权重并通过双向视觉–动作注意力交互。

项目页报告统一训练相较单模式训练，在四个数据集的 Perceive 相机空间 MPJPE 平均降低 23.46%，Predict 的 PSNR 平均提高 0.782 dB；Plan 的五项 VLM 评分也有提升。Plan 评分是代理评价，不等同于真机成功率。

## 开源核查

- **代码：待发布。** 项目页标注 “Code forthcoming”，截至入库日未提供公开 GitHub 仓库链接。
- **数据：** 项目页未列公开数据集下载链接。
- **机器人演示：** 生成的人手轨迹经重定向后用于机器人执行；页面明确说明演示是开环执行，执行时不接收新观测进行纠正。
- 页面提供论文与补充视频，可用于查看方法和定性演示；这些资源不代表训练/部署代码已开放。

## 对 Wiki 的映射

- [P³-WAM 实体页](../../wiki/entities/paper-p3-wam-egocentric-dexterous.md)
- [论文归档](../papers/p3_wam_arxiv_2610_22363.md)
