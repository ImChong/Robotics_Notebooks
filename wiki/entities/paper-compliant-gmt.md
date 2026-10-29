---
type: entity
tags: [paper, humanoid, whole-body-control, compliant-control, motion-tracking, unitree-g1]
status: complete
updated: 2026-10-27
project_id: compliant-gmt-residual-reference-adaptation
arxiv: "2610.22119"
project: https://venki1899.github.io/c-gmt/
summary: "C-GMT 在冻结的通用运动跟踪器前加入残差参考适配器，以本体感知实现全身柔顺响应。"
related:
  - ./paper-gmt.md
  - ./paper-oclo-online-posture-compliant-loco-manipulation.md
  - ../concepts/whole-body-control.md
sources:
  - ../../sources/papers/compliant_gmt_arxiv_2610_22119.md
  - ../../sources/sites/compliant-gmt.md
---

# Compliant GMT：残差参考适配的全身柔顺运动跟踪

**一句话定义**：C-GMT 为预训练通用运动跟踪器加入参考动作残差，使机器人在跟踪多样动作时对外力产生全身柔顺响应。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| C-GMT | Compliant General Motion Tracking | 柔顺通用运动跟踪框架 |
| GMT | General Motion Tracking | 通用运动跟踪 |
| IK | Inverse Kinematics | 逆运动学，用于教师的柔顺参考求解 |
| G1 | Unitree G1 humanoid robot | 真机实验平台 |

## 核心机制

- **教师：** 使用真实接触力和指定刚度，经柔顺 IK 求出修正后的参考；冻结的抗力扰动跟踪器执行该参考。
- **学生：** 根据参考动作及本体感知历史预测参考残差，并通过动作匹配学习教师行为；部署时不依赖教师的真实接触力输入。
- **设计取舍：** 调整跟踪器的参考而非直接替换整个跟踪策略，试图保留预训练跟踪器的动作泛化能力，同时获得柔顺响应。

## 评测与局限

论文报告仿真与 G1 真机的跟踪精度、柔顺性、安全性和鲁棒性实验；项目页展示舞蹈、遥操作、插入与开门等场景，并演示可调刚度。演示不等于所有真实场景均经过量化验证。

截至 2026-10-27，项目页标注代码待发布，未列出可运行代码或数据与权重下载入口；独立复现仍需等待材料公开。

## 与其他工作对比

| 工作 | 柔顺来源 | 与 C-GMT 的差异 |
|------|----------|-----------------|
| [GMT](./paper-gmt.md) | 无显式柔顺；以统一策略刚性跟踪参考 | C-GMT 正是要解决这类通用跟踪器「抵抗外力」的问题；它冻结跟踪器，只在参考端加残差 |
| [OCLO](./paper-oclo-online-posture-compliant-loco-manipulation.md) | 力反馈驱动的参考偏移 | 同样修改参考而非策略；OCLO 从双末端目标在线生成姿态、不依赖人体动作数据，C-GMT 面向任意人体动作跟踪，并用学生从本体感知历史推断外力，无需力传感 |

关键取舍：改参考比重训跟踪策略便宜，也保留了预训练跟踪器的泛化；代价是柔顺能力受限于学生从本体感知推断接触的准确度。

## 结论

C-GMT 提供了把柔顺性加入通用动作跟踪器的模块化思路。

- 关注残差修正的参考是否保持动作可行，以及外力作用下的跟踪误差。
- 比较期望刚度与实际响应，不能只看无接触时的跟踪精度。
- 真机部署应结合接触场景验证安全性与鲁棒性；项目演示不能替代具体任务评测。

## 源码运行时序图

不适用：项目页目前仅标注代码待发布。

## 关联页面

- [GMT：通用全身动作跟踪](./paper-gmt.md)
- [OCLO：在线姿态与柔顺移动操作](./paper-oclo-online-posture-compliant-loco-manipulation.md)
- [全身控制](../concepts/whole-body-control.md)

## 参考来源

- [论文资料归档](../../sources/papers/compliant_gmt_arxiv_2610_22119.md)
- [项目页归档](../../sources/sites/compliant-gmt.md)

## 推荐继续阅读

- [论文原文](https://arxiv.org/abs/2610.22119)
- [项目演示](https://venki1899.github.io/c-gmt/)
