# CoDance 项目主页（General Robotics Lab）

> 项目页来源归档

- **标题：** CoDance — Learning Reactive and Compliant Human-Humanoid Interaction from Video
- **类型：** site / research project
- **URL：** <https://generalroboticslab.com/CoDance>
- **所属机构：** General Robotics Lab, Duke University
- **论文：** <https://arxiv.org/abs/2610.05324> — [论文归档](../papers/codance_arxiv_2610_05324.md)
- **代码：** <https://github.com/generalroboticslab/codance> — [源码归档](../repos/codance.md)
- **演示：** <https://youtu.be/ufOYmNZvLVU>
- **数据集与检查点：** <https://huggingface.co/datasets/GeneralRoboticsLab/CoDance>
- **入库日期：** 2026-10-11
- **一句话说明：** 以单段双人舞视频训练可跟随舞伴并对双手接触作顺应响应的人形机器人策略。

## 开源核查（2026-10-11）

| 入口 | 状态 |
|---|---|
| 研究项目页 | 已公开，列出论文、演示、数据与部署代码入口。 |
| 训练 / 评测 / 增强源码 | 已公开，见官方 GitHub 仓库 README。 |
| 真机部署 | 已公开，仓库含 Unitree G1 的 simple_deploy 说明与运行代码。 |
| 数据 / 检查点 | 官方 README 链接 Hugging Face 数据集；README 描述下载约 850 MB，含两类训练 clip pool、七个检查点和 ONNX 导出。 |
| 许可 | 根项目 Apache-2.0；仓库同时包含具有不同许可的第三方部分，需逐项核对。 |

**开放程度：已开源（有第三方许可边界）。** 官方 GitHub README 列出训练、评测、顺应增强与真机部署代码；Hugging Face 下载入口由 README 指向。本站未独立核验 HF 文件清单，数据内容描述以仓库 README 为依据。

## 项目页内容要点

- CoDance 将一段双人舞视频转成机器人参考与移动舞伴，再通过结构化双手外力做多连杆顺应增强。
- 训练策略在仿真中跟随舞伴并响应互动；官方项目介绍报告 G1 真机持续双手接触舞蹈。
- 实机使用 Vicon 舞伴追踪；这使部署结果依赖动作捕捉场地与标定，不是仅靠视频摄像头的在线感知演示。

## 对 wiki 的映射

- [CoDance 实体页](../../wiki/entities/paper-codance.md)
- [论文来源](../papers/codance_arxiv_2610_05324.md)
- [官方仓库来源](../repos/codance.md)
