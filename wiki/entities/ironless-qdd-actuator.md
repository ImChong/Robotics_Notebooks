---
type: entity
tags: [hardware, actuator, qdd, open-source, cycloidal, bldc, 3d-print]
status: complete
updated: 2026-07-25
related:
  - ../comparisons/open-source-qdd-actuator-projects.md
  - ./internal-cycloidal-actuator.md
  - ./cycloidal-quasi-direct-drive-actuator.md
  - ./opentorque-actuator.md
  - ./berkeley-humanoid-lite.md
  - ../../roadmap/depth-torque-motor-design.md
  - ../queries/actuator-drive-chain-selection-loop.md
sources:
  - ../../sources/repos/ironless_qdd_actuator.md
  - ../../sources/sites/cadenkraft_ironless_cycloidal_planetary_actuator.md
  - ../../sources/personal/open_source_qdd_actuator_learning_curator.md
summary: "Caden Kraft Ironless QDD：Halbach 无铁芯转子 + 成品定子 + 3D 打印摆线—行星减速 + 集成驱动/磁编；BOM 约 $40–70，报告静态保持约 29.4 N·m；学习低成本电磁与打印传动，须区分保持力矩≠连续动态力矩。"
---

# Ironless QDD Actuator（无铁芯转子摆线—行星执行器）

## 一句话定义

**Ironless QDD Actuator**（[CKraft11/Ironless-QDD-Actuator](https://github.com/CKraft11/Ironless-QDD-Actuator)，项目长文 [cadenkraft.com](https://cadenkraft.com/ironless-cycloidal-planetary-actuator/)）是低成本、**全定制件可 3D 打印** 的准直驱关节：成品定子 + **Halbach 无铁芯转子** + **摆线—行星** 组合减速 + 集成驱动与磁编。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| QDD | Quasi-Direct Drive | 准直驱，低减速比、高背驱动性的作动方案 |
| BOM | Bill of Materials | 物料清单，硬件零部件列表 |
| FEMM | Finite Element Method Magnetics | 开源 2D 电磁有限元工具 |
| BLDC | Brushless DC Motor | 无刷直流电机 |
| CAD | Computer-Aided Design | 计算机辅助设计，硬件结构建模 |

## 为什么重要

- 在「电机本体也开源」一类里，它把 **无铁芯 / Halbach 转子取舍** 用 FEMM 对照写清楚，适合学磁路直觉。
- BOM 约 **$40（执行器）/ $70（含控制器）**，门槛极低；与 [Internal Cycloidal](./internal-cycloidal-actuator.md)（~$384、含机加件）形成成本—成熟度对照。
- 明确示范：**静态保持力矩 ≠ 连续行走力矩**——读开源执行器指标时的硬课。

## 核心信息

| 项 | 内容 |
|----|------|
| 转子 | **Ironless + Halbach**（无铁背板）；作者用 FEMM 对比有/无铁、Halbach/常规阵列 |
| 定子 | 成品定子采购 |
| 减速 | 3D 打印 **摆线—行星** 组合；作者称零背隙叙事 |
| 驱动 | 集成驱动；仓库含 **MKS XDrive / ODrive 配置** |
| 传感 | 磁编码器 |
| 力矩指标 | 报告 **~29.4 N·m 静态保持**（勿当连续/冲击额定） |
| 许可 | README 写 **MIT**（需署名 Caden Kraft） |
| 资产 | CAD、FEMM、BOM；大文件走 **Git LFS**（亦镜像 MakerWorld） |

## 核心原理

```mermaid
flowchart LR
  stator["成品定子"]
  rotor["3D 打印转子\nHalbach 无铁芯"]
  gear["摆线—行星\n3D 打印"]
  out["输出"]
  drv["集成驱动 + 磁编"]
  stator --- rotor --> gear --> out
  drv --> stator
```

- **为何无铁芯：** 转子全打印时避免铁背板；Halbach 把磁通压向气隙侧，用 FEMM 量化相对有铁方案的扭矩代价。
- **材料：** 作者强调工程塑料（PA6-GF/CF、PET-CF 等）；**PLA 转子会因磁力蠕变**；转子需 **100% 填充**、高刚度低蠕变。

## 工程实践

| 项 | 建议 |
|----|------|
| 克隆 | 先装 Git LFS；带宽不够用 MakerWorld 镜像 |
| 打印 | 转子工程塑料 + 100% 填充；齿轮可用 CF-Core 类（作者称未充分验证） |
| 装配 | 行星架需 M3 攻丝；减速器内涂锂基脂 |
| 读指标 | 只把 29.4 N·m 当**保持能力上界**，自行测连续温升与动态力矩 |

## 局限与风险

- **~30 N·m 是静态保持**，不是连续动态输出、额定或冲击力矩，更不是人形行走可长期使用力矩。
- 个人 DIY 验证；缺公开长寿命/热循环台架报告。
- 全塑料传动在高冲击腿足上需自行评估齿瓣与蠕变。

## 关联页面

- [开源 QDD 执行器项目对比](../comparisons/open-source-qdd-actuator-projects.md)
- [Internal Cycloidal Actuator](./internal-cycloidal-actuator.md)
- [Cycloidal QDD（Jeong）](./cycloidal-quasi-direct-drive-actuator.md)
- [OpenTorque Actuator](./opentorque-actuator.md)
- [执行器驱动链选型闭环](../queries/actuator-drive-chain-selection-loop.md)

## 参考来源

- [sources/repos/ironless_qdd_actuator.md](../../sources/repos/ironless_qdd_actuator.md)
- [sources/sites/cadenkraft_ironless_cycloidal_planetary_actuator.md](../../sources/sites/cadenkraft_ironless_cycloidal_planetary_actuator.md)
- [开源 QDD 执行器学习策展](../../sources/personal/open_source_qdd_actuator_learning_curator.md)

## 推荐继续阅读

- 项目长文：<https://cadenkraft.com/ironless-cycloidal-planetary-actuator/>
- 仓库：<https://github.com/CKraft11/Ironless-QDD-Actuator>
- MakerWorld 镜像（LFS 备援）：README 内链接
