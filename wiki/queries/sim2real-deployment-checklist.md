---
type: query
tags: [sim2real, deployment, checklist, real-robot, debugging, locomotion]
status: complete
summary: 真机部署 RL 策略前后的完整检查清单，覆盖训练端、部署端、调试端三个阶段。
sources:
  - ../../sources/papers/sim2real.md
related:
  - ../concepts/sim2real.md
  - ../concepts/domain-randomization.md
  - ../tasks/locomotion.md
  - ../comparisons/sim2real-approaches.md
---

# Sim2Real 真机部署清单

> **Query 产物**：本页由问题「真机部署 RL 策略前后要检查什么？」触发。
> 综合来源：[Sim2Real](../concepts/sim2real.md)、[Domain Randomization](../concepts/domain-randomization.md)

## 一句话定义

把仿真训练好的策略部署到真机，需要在**训练端、部署端、调试端**三个阶段依次排查问题，避免策略在真机上失效或损坏硬件。

---

## 阶段一：训练端准备

| 检查项 | 典型范围 | 通过标准 |
|-------|---------|---------|
| 质量/惯量随机化 | ±20% | 仿真中性能稳定 |
| 关节摩擦/阻尼随机化 | ±30% | 无明显策略退化 |
| 延迟随机化 | 0~40ms | 覆盖真实延迟上限 |
| 观测噪声 | 参照真机规格 | 噪声量级与真机一致 |

- [ ] 关节力矩限制在仿真中强制执行（与真机规格一致）
- [ ] 策略输出频率与真机控制频率匹配（50Hz 或 100Hz）
- [ ] 观测归一化参数在导出时固定（不依赖运行时统计）

## 阶段二：部署端检查

- [ ] 模型文件版本与训练代码一致
- [ ] 观测空间维度和顺序与真机接口匹配（逐字段核对）
- [ ] 动作空间映射：仿真关节顺序 ↔ 真机 CAN ID 顺序
- [ ] 推理延迟测试 < 5ms
- [ ] 初始姿态与仿真初始化姿态一致（误差 < 5°）

## 阶段三：调试端常见问题

| 现象 | 可能原因 | 排查方向 |
|-----|---------|---------|
| 上机立刻摔倒 | 观测顺序错误 | 核对 obs/action 映射 |
| 原地抖动 | 控制频率不匹配 | 检查 PD 增益 |
| 向一侧漂移 | IMU 安装偏差 | 校准 IMU offset |
| 步伐极小 | 命令归一化错误 | 打印原始策略输入 |
| 关节力矩饱和 | 力矩映射错误 | 核查 scale 因子 |

## 参考来源

- Kumar et al., *RMA: Rapid Motor Adaptation* (2021)
- [Sim2Real 概念页](../concepts/sim2real.md)
- [Domain Randomization](../concepts/domain-randomization.md)

## 关联页面

- [Sim2Real](../concepts/sim2real.md)
- [Domain Randomization](../concepts/domain-randomization.md)
- [Locomotion](../tasks/locomotion.md)
- [Sim2Real 方法对比](../comparisons/sim2real-approaches.md)
- [Query：RL 策略真机调试 Playbook](./robot-policy-debug-playbook.md) — 系统排障流程的完整版本
