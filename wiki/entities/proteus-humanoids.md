---
type: entity
project_id: proteus-one-policy-all-humanoids
tags: [humanoid, motion-tracking, cross-embodiment, simulation]
status: draft
updated: 2026-10-28
project: https://proteushumanoids.vercel.app/
related:
  - ./paper-x-wbc.md
  - ../methods/motion-retargeting-gmr.md
  - ../concepts/whole-body-tracking-pipeline.md
sources:
  - ../../sources/sites/proteus-humanoids.md
summary: "Proteus 在仿真中以一套权重和关节级本体描述跟踪跨 60 种人形机器人的动作。"
---

# Proteus：一套策略跟踪多种人形机器人的动作

## 一句话定义

**Proteus 使用共享关节布局、本体参数描述和联合训练，让一套策略权重在多种人形机器人上跟踪重定向的人体动作。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| WBT | Whole-Body Tracking | 全身动作跟踪 |
| GMR | General Motion Retargeting | 人体动作重定向工具 |
| ONNX | Open Neural Network Exchange | 模型交换格式 |
| DoF | Degree of Freedom | 自由度 |

## 核心方法

1. 将不同机器人的关节映射到共同布局，使参考动作有统一语义。
2. 向策略提供关节描述和身体比例，而非机器人名称；页面将不同身体视为更大范围的关节级随机化。
3. 在仿真中联合训练，页面称每种身体在每次更新中具有相同权重；同一组策略权重服务所有本体。
4. LAFAN1 动作经 [GMR](../methods/motion-retargeting-gmr.md) 重定向；在 Isaac Lab 训练，导出 ONNX 后在 MuJoCo 评测。

## 评测与泛化

项目页展示来自 25 家制造方的 60 种人形机器人，具有 18–31 个驱动关节、0.28–1.02 m 髋高。留出动作评测以 20 秒片段完整跟踪且全程平均身体误差小于 0.5 m 为成功条件。两个仿真器的片段数量及前 0.2 秒处理不同，不能直接把图中结果视作严格同条件对比。

较早的 45 机器人模型对部分未训练、与训练机器人相近的本体，在 MuJoCo 的留出片段上达到 40–50% 成功率；页面也明确指出形态差异较大的未见本体仍会摔倒。**已训练本体上的动作泛化与未见本体零样本泛化应分开统计。**

## 工程实践与局限

- 这是项目页自报的仿真结果；页面展示 MuJoCo 渲染与浏览器仿真，不能据此推断真机效果。
- 截至 2026-10-28，项目页标注论文和代码即将发布，未给出源码链接。网页演示会加载 ONNX 策略，但训练配方、完整评测及可复现性仍待论文与代码核实。
- 新本体验收应分机器人、动作和模拟器记录成功率，先核对关节映射、限位、动力学与参考动作；网页没有证明任意新人形都可直接运行。
- 页面称可原则上扩展至四足或轮式机器人，但未给出对应验证结果。

## 关联页面

- [X-WBC](./paper-x-wbc.md)：另一条跨人形全身控制路线，需按各自评测和本体范围比较。
- [全身动作跟踪流水线](../concepts/whole-body-tracking-pipeline.md)：理解参考轨迹、策略和部署之间的接口。
- [动作重定向（GMR）](../methods/motion-retargeting-gmr.md)：人体参考动作的上游转换。

## 参考来源

- [Proteus 项目页归档](../../sources/sites/proteus-humanoids.md)

## 推荐继续阅读

- [Proteus 官方项目页及网页演示](https://proteushumanoids.vercel.app/)
