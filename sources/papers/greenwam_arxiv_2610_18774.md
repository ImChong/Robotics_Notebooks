# GreenWAM 论文来源归档

## 元数据
- **题目：** GreenWAM: A World Foundation Model for Robotics
- **作者：** Gennady Sidorov、Valerii Kornilov、Denis Grankin、Polina Fedotova、Aleksey Postnikov
- **机构：** Brain Team、Sber Robotics Center
- **预印本：** https://arxiv.org/abs/2610.18774
- **项目页：** https://greenvla.github.io/GreenWAM/
- **代码：** https://github.com/greenvla/GreenWAM
- **权重：** https://huggingface.co/SberRoboticsCenter/GreenWAM
- **提交日期：** 2026-10-13
- **入库日期：** 2026-10-15
- **一句话说明：** GreenWAM 将跨 20 类机器人本体的共享关节动作表示、世界模型训练和策略学习合入同一模型，并发布评测代码与四个基准权重。

## 论文要点
论文提出一个同时承担动作条件世界预测与策略生成的机器人基础模型。它把各本体的关节映射到共享解剖动作空间，并联合训练视觉、动作条件世界预测与 world-action 模式；面向策略适配时，将动作专家从生成器中拆出。

论文报告 GreenWAM 在 WorldArena 1.0 的轨迹、深度和 3D 预测表现居前；在 RoboTwin 2.0、LIBERO-plus、CALVIN 与 LIBERO 上达到有竞争力的策略结果，并称经适配后在真机人形上优于所比较的 π₀.₅ 与 Cosmos 3 Edge 策略。

## 对 wiki 的映射
- [GreenWAM 实体页](../../wiki/entities/paper-greenwam.md)：统一动作空间、训练模式、动作专家、基准结果与复现边界。
- [世界–动作模型](../../wiki/concepts/world-action-models.md)：作为显式世界预测与动作策略合一的实例。

## 开源状态
论文发布时，官方项目仓库提供评测代码、策略服务端、benchmark client 与四个基准 checkpoint；完整复现仍依赖各 benchmark 的独立环境、资产和模型组件。详见 [官方仓库归档](../repos/greenwam.md)。
