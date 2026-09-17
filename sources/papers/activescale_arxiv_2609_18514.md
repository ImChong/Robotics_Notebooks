# ActiveScale（arXiv:2609.18514）

> 来源归档（paper）

- **标题：** ActiveScale: Scaling Active Perception for Robots across Model, Data, and Hardware
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2609.18514>
- **PDF：** <https://arxiv.org/pdf/2609.18514>
- **项目页：** <http://active-scale.github.io/>
- **入库日期：** 2026-09-17
- **一句话说明：** 在 π₀.5 上叠加历史帧 + 相机 pose token 监督的主动感知 VLA；1000 小时人机 mid-training + AMP 移动操作平台；五类真机任务 mean SR 30%→70%。

## 开源状态

- **待发布**（步骤 2.5 核查，2026-09-17）：项目页未列 GitHub / 权重链接。

## 核心摘录

1. **模型：** 当前帧 + 间隔 16 帧采样的历史帧；每帧 camera token + 9D 相机头监督平移/旋转/FOV；推理时移除相机头。
2. **数据：** 1000 小时 egocentric + 机器人 mid-training（EgoLive/EgoVerse/EgoSuite + AgiBot World/RoboCOIN/AMP）。
3. **硬件 AMP：** AgileX Cobot-Magic 扩展第三臂作主动相机 + 双臂 + 移动底座；Quest 2 单操作员遥操作。
4. **评测：** 五类任务各 150 demo、20 rollouts；mean SR 30.0%→70.0%，mean TP 41.6%→78.4%；50 动作块 RTX 4090 264.5 Hz。

**对 wiki 的映射**

- [paper-activescale](../../wiki/entities/paper-activescale.md)
- [9 篇技术地图](../../wiki/overview/perception-action-transfer-9-papers-technology-map.md)
