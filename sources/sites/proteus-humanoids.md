# Proteus 官方项目页

> 来源归档（ingest）

- **标题：** Proteus: One Policy, All Humanoids
- **类型：** official project page
- **项目页：** <https://proteushumanoids.vercel.app/>
- **网页演示：** <https://proteushumanoids.vercel.app/demo/>
- **页面核查：** 2026-10-28
- **作者：** 项目页标注 Anonymous authors
- **论文 / 代码：** 项目页均标注 coming soon，未提供论文或源码链接；网页演示加载 ONNX 策略，但不等同于训练代码开放。
- **一句话说明：** 在仿真中联合训练单一动作跟踪策略，以关节描述和身体比例适配 60 种人形机器人。

## 页面要点

- 60 种机器人来自 25 家制造方，18–31 个驱动关节，髋高 0.28–1.02 m；使用相同权重跟踪人类动作。
- 动作数据为 LAFAN1，经 GMR 重定向；在 Isaac Lab 训练，并在 MuJoCo 使用导出的 ONNX 策略评估。
- 评测为留出的 20 秒动作片段，要求全程平均身体误差小于 0.5 m；不同模拟器评测片段数和片段起始处理不同。
- 较早的 45 机器人模型对部分未训练的近似本体有 40–50% 的片段成功率；页面称与训练本体差异大的机器人仍会摔倒。

## 对 wiki 的映射

- [Proteus 实体页](../../wiki/entities/proteus-humanoids.md)
