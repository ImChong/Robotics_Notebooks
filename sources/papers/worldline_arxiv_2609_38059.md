# WorldLine：面向机器人操作的动作驱动视觉仿真

> 来源归档（paper）

- **标题：** WorldLine: Action-Driven Visual Simulation for Robotic Manipulation
- **类型：** arXiv 论文（cs.RO）
- **作者：** Shenghe Zheng、Wenbo Li、Jiyao Zhang、Bin Xia、Haoyang Huang、Nan Duan、Jiaya Jia
- **机构：** 香港科技大学、Joy Future Academy、北京大学、香港中文大学
- **arXiv：** <https://arxiv.org/abs/2609.38059>
- **PDF：** <https://arxiv.org/pdf/2609.38059>
- **项目页：** <https://zhengsh123.github.io/WorldLine/>
- **代码仓：** <https://github.com/Zhengsh123/WorldLine>（仓库当前不含训练或推理实现）
- **首次公开：** 2026-09-29
- **入库日期：** 2026-10-15
- **一句话说明：** 将跨本体操纵视频动力学学习与图像空间动作接地拆成三阶段，用视频 rollout 评估策略候选并支持具身规划。

## 核心摘录（MVP）

### 1) 先用无动作标注视频学机器人–物体动态

- **链接：** <https://arxiv.org/abs/2609.38059>
- **摘录要点：** Stage I 从 Cosmos3-Nano 初始化，在超过 10,000 小时的无动作条件机器人视频上预训练。视频覆盖 10 余种本体、3,500 余项操纵任务；该阶段不把已有动作标注作为输入，让昂贵的动作监督不再限制动态先验的规模。
- **对 wiki 的映射：** [WorldLine 实体页](../../wiki/entities/paper-worldline-action-driven-visual-simulation.md)；[生成式世界模型](../../wiki/methods/generative-world-models.md)

### 2) 将异构机器人控制投影到共享图像空间

- **链接：** <https://arxiv.org/abs/2609.38059>
- **摘录要点：** Stage II 用机器人运动学与相机标定，把末端位置、深度、Rot6D 姿态和夹爪状态投影为每个视角的九通道 action map。同步多视角、约 200 小时失败轨迹与关系正则共同提升跨视角和失败交互预测；数据表报告约 2,500 小时动作轨迹。
- **对 wiki 的映射：** [WorldLine 实体页](../../wiki/entities/paper-worldline-action-driven-visual-simulation.md)；[世界模型功能分类](../../wiki/concepts/functional-taxonomy-world-models.md)

### 3) 用 rollout 做离线评估与候选动作筛选

- **链接：** <https://arxiv.org/abs/2609.38059>
- **摘录要点：** Stage III 蒸馏为因果分块自回归模型，每块用四步去噪生成 16 个 RGB 帧；129 帧生成从 90 秒降到 21 秒（约 4.3×）。在 RoboTwin 与 AgiBot 成功/失