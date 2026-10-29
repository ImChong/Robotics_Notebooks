# Kine2Go: Kinematic dataset for the Unitree Go2 robot with diverse gaits and motions（arXiv:2606.14433）

> 来源归档（ingest）

- **标题：** Kine2Go: Kinematic dataset for the Unitree Go2 robot with diverse gaits and motions
- **类型：** paper / quadruped locomotion / motion retargeting / imitation learning / dataset
- **arXiv：** <https://arxiv.org/abs/2606.14433>
- **项目页：** <https://nomagiclab.github.io/kine2go-pipeline/>
- **代码：** <https://github.com/nomagiclab/kine2go-pipeline>
- **数据：** <https://huggingface.co/datasets/MIMUW-Robotics/kine2go>
- **作者：** Władysław Pałucki、Paweł Siwak、Krzysztof Ciebiera、Marek Cygan
- **机构：** University of Warsaw（华沙大学）
- **发表日期：** 2026-06-12
- **一句话说明：** 将狗、马及合成四足动作重定向至 Unitree Go2，以 Genesis 中的 PPO 模仿策略采集多样步态轨迹，发布 40 个参考动作及 800 条策略 rollout。

## 摘要级要点

- **数据规模：** 40 段参考动作；每段对应一个 PPO 模仿策略与 20 条扰动 rollout，共 800 条。
- **生成流程：** 源四足动作 → Go2 形态重定向 → RL 跟踪训练 → rollout 采集与过滤。
- **数据内容：** Go2 运动学参考轨迹、策略配置与 rollout 数据（含状态及电机命令），并提供渲染视频。
- **平台：** Unitree Go2（12 个驱动关节）、Genesis 仿真器；数据采集频率为 60 Hz。
- **用途：** 可用于四足运动模仿、行为基础模型训练/正则化，以及动作重定向评估。
- **许可：** 数据按上游来源分别许可，见数据集 LICENSE 与每段 clip.json；不可把整包视作单一许可。

## 对 wiki 的映射

- 主实体：[Kine2Go 数据集](../../wiki/entities/paper-kine2go.md)
- 关联概念：[Motion Retargeting](../../wiki/concepts/motion-retargeting.md)

## 参考来源（原始）

- [arXiv:2606.14433](https://arxiv.org/abs/2606.14433)
- [Kine2Go 项目页](https://nomagiclab.github.io/kine2go-pipeline/)
- [Hugging Face 数据集卡](https://huggingface.co/datasets/MIMUW-Robotics/kine2go)
- [官方 pipeline 仓库](https://github.com/nomagiclab/kine2go-pipeline)
