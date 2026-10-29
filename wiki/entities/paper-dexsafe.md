---
type: entity
project_id: dexsafe
tags: [paper, manipulation, dexterous-manipulation, reinforcement-learning, sim2real, safety, berkeley]
status: complete
updated: 2026-10-29
arxiv: "2610.37652"
project: https://ian-chuang.github.io/dexsafe/
summary: "DexSafe 在仿真训练 LEAP 手内位姿到达策略，并通过可调保守度的残差安全过滤减少真机掉落。"
related:
  - ../tasks/manipulation.md
  - ../methods/in-hand-reorientation.md
sources:
  - ../../sources/papers/dexsafe_arxiv_2610_37652.md
  - ../../sources/sites/dexsafe.md
---

# DexSafe：灵巧手位姿到达与安全过滤

**DexSafe** 在仿真训练灵巧手移动手中物体到目标位姿，并以学习式残差过滤器降低掉落风险。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|---|---|---|
| RL | Reinforcement Learning | 仿真中训练任务策略 |
| DoF | Degrees of Freedom | 描述物体位置和姿态的自由度 |
| OOD | Out of Distribution | 分布外目标或扰动 |

## 核心机制

| 项目 | 内容 |
|---|---|
| 机构 | 加州大学伯克利分校、加州大学戴维斯分校 |
| 训练 | 生成抓取数据提供多样初始状态和目标，配合自适应摩擦课程；无需人类示范 |
| 感知 | 特权教师蒸馏到状态学生和深度学生，真机分别使用动作捕捉或腕部深度相机 |
| 安全 | ResSafe 对任务动作施加学习式残差，保守度可调；目标是减少物体掉落 |
| 范围 | LEAP 手；项目页说明每个物体对应一个策略，并非单策略泛化到全部物体 |

## 实验与评测

真机对四个物体、六种手腕朝向执行 4,800 个未见目标，共 13.3 小时。按 2 cm 和 0.4 rad 阈值，状态策略达到 95.2%；加过滤器后达到 91.5%，掉落从 16 次降至 4 次。该结果针对状态策略和所测配置，不能直接推广到深度策略或任意物体。项目页展示深度策略在刀和酒杯上的结果明显更困难。

## 结论

任务策略与安全过滤分开训练，可在目标成功率和掉落风险之间调节。

1. 复现先核对每物体独立策略与目标阈值。
2. 动捕与深度输入的评测结果应分开报告。
3. 安全过滤降低掉落但会牺牲部分到达率，需按应用调整保守度。

## 源码运行时序图

**不适用：** 2026-10-29 项目页标注代码待发布；已有浏览器演示，训练及部署源码入口尚未列出。

## 局限与工程实践

系统辨识对零样本迁移有作用；结果基于四个物体，无法据此断言泛化到所有物体。相机遮挡、动态干扰和手腕姿态均需在目标任务上重新评估。

## 关联页面

- [操作任务](../tasks/manipulation.md)
- [手内重定向](../methods/in-hand-reorientation.md)

## 参考来源

- [论文归档](../../sources/papers/dexsafe_arxiv_2610_37652.md)
- [项目页归档](../../sources/sites/dexsafe.md)

## 推荐继续阅读

- [项目演示与实验表格](https://ian-chuang.github.io/dexsafe/)
