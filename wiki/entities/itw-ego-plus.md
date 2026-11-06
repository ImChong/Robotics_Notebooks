---
type: entity
tags: [dataset, humanoid, motion-capture, egocentric, embodied-ai]
status: draft
updated: 2026-11-06
project_id: itw-ego-plus
project: https://modalitynet.com/
related:
  - ./paper-hiphi.md
  - ../comparisons/humanoid-reference-motion-datasets.md
  - ../methods/imitation-learning.md
sources:
  - ../../sources/sites/itw-ego-plus.md
original:
  - ../../sources/sites/itw-ego-plus.md
summary: "Noitom Robotics 的 ITW-Ego Plus 将真实环境中的第一视角视频与全身运动及其他模态结合；数据许可和规模待核验。"
---

# ITW-Ego Plus：真实环境第一视角与全身运动数据

ITW-Ego Plus 是 Noitom Robotics 在 ModalityNet 数据体系中提出的真实环境人体多模态数据类别，结合第一视角视频与全身运动。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| ITW | In the Wild | 非实验室约束下的真实环境 |
| RGB-D | Red Green Blue and Depth | 彩色图像与深度信息 |
| VLA | Vision-Language-Action | 视觉、语言和动作建模 |
| MoCap | Motion Capture | 人体运动采集 |

## 数据速查

| 维度 | 截至 2026-11-06 的核查结果 |
|------|------------------------------|
| 规模 | 未公开；官方目录需登录，未读取 |
| 模态 | 第一视角视频 + 全身运动；官方公告另列腕部、外视、触觉、声音等扩展模态（因采集配置而异） |
| 许可证 | 未确认；HiPHI 的研究开放许可不能直接套用到本数据集 |
| 重定向就绪度 | 未验证；人体运动到机器人的重定向与同步误差需另行评估 |

## 采集流程

```mermaid
flowchart LR
  Person["真实环境中的人和任务"] --> Sol["Sol 第一视角采集"]
  Person --> Jupiter["Jupiter 可穿戴运动采集"]
  Sol --> Data["第一视角与全身运动数据"]
  Jupiter --> Data
  Data --> Use["数据浏览与机器人学习研究"]
```

官方称 Jupiter 使用少量穿戴设备采集全身运动，不需要动捕服。官方公告还列出腕部、外视、触觉、声音等扩展模态；具体模态取决于采集配置，不能假设所有样本均具备。

更广义的 ITW 技术文章介绍：利用视觉与稀疏身体感知重建手腕和手部运动；部分带侧视 RGB-D 的数据可进行全身场景重建。该描述不等于每条 ITW-Ego Plus 数据都经过相同处理。

## 研究价值与限制

- 第一视角提供动作发生时的视觉上下文，全身运动补充相机画面之外的身体信息，可为人体行为建模和机器人学习提供参考。
- 人体运动不是机器人控制指令；具体重定向、同步误差及训练有效性需要数据和实验进一步验证。
- 截至 2026-11-06，官方目录需要登录；公开资料未确认该数据集的完整下载方式、规模、标注字段及许可。HiPHI 研究开放许可不能直接视为本数据集的许可。
- **源码运行时序图：不适用。** 本次未找到本项目可运行的官方采集或训练实现；也未核实交互浏览器源码。
- 官方公告和公开技术文章已核查，但登录后的数据目录与样本尚未读取，因此此页为待完善草稿。

## 关联页面

- [HiPHI](./paper-hiphi.md)：同属 ModalityNet 的高精度人体与物体交互数据，采集设置和开放范围不同。
- [人形参考运动与操作数据集选型](../comparisons/humanoid-reference-motion-datasets.md)
- [模仿学习](../methods/imitation-learning.md)

## 推荐继续阅读

- [ModalityNet 官方主页](https://modalitynet.com/)
- [Noitom 官方技术文章](https://huggingface.co/blog/noitomrobotics/art-of-modalities)

## 参考来源

- [ITW-Ego Plus 官方资料归档](../../sources/sites/itw-ego-plus.md)
