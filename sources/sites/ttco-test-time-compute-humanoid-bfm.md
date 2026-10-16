# TTCO 项目页：bfmttc.github.io

> 来源归档（site；2026-10-16 已核验源码链接）

- **标题：** Test-Time Compute for Humanoid Behavior Foundation Models
- **类型：** site / project-page
- **URL：** <https://bfmttc.github.io/>
- **论文：** [arXiv:2610.14896](https://arxiv.org/abs/2610.14896)
- **代码：** <https://github.com/Gamma-Robot/bfm_ttc>
- **数据 / 权重：** 页面未提供单独下载链接；仓库 README 要求另行准备 HEFT-PMG checkpoint 与 G1 robot assets。
- **入库日期：** 2026-10-14；源码核验更新：2026-10-16
- **沉淀到 wiki：** [TTCO 统一论文实体页](../../wiki/entities/paper-ttco-test-time-compute-humanoid-bfm.md)

## 一句话摘要

官方项目页介绍 **Test-Time Command Optimization（TTCO）**：通过测试时优化冻结人形行为基础模型（BFM）的输入命令，在不重训 BFM 的前提下提高 motion tracking fidelity；页面同时展示三种 BFM 的 tracking 误差、预算扩展和高层策略真机评测。

## 项目页核验记录

- **作者：** Weiji Xie、Junsong Wu、Zibo Zhou、Yueru Chen、Xingyi Wang、Zhenhua Xiong、Weinan Zhang、Chenjia Bai。
- **机构：** Institute of Artificial Intelligence (TeleAI), China Telecom；Shanghai Jiao Tong University；Zhejiang University；Gamma Robotics。
- **源码状态：已开源。** 页面 Resources 区的 Code 按钮直接连到 [Gamma-Robot/bfm_ttc](https://github.com/Gamma-Robot/bfm_ttc)。
- **权重与数据边界：** 项目页没有独立权重或数据下载入口。代码虽已公开，但运行示例需要模型 checkpoint 和 G1 资产。

## 页面主要结果

### Motion tracking（仿真）

项目页称在 AMASS 的 **12,273** 条参考动作上评估 HEFT、SONIC 与 TWIST2；平均全局身体位置误差如下（厘米）：

| BFM | 参考动作直接作命令 | TTCO 优化命令 |
|-----|-------------------:|----------------:|
| HEFT | 27.1 | 5.7 |
| SONIC | 61.5 | 12.7 |
| TWIST2 | 109.5 | 17.3 |

总体误差下降 **79–84%**。项目页说明真机片段是在仿真优化命令上的回放，没有在线 replanning。

### 计算预算

项目页展示 surrogate cost model 的预算分配分析：平均 cost regret 为 **4.02%**；K=256 提升至 K=8192 时 cost 降低 **58.0%**。

### 示范重用与高层策略

- 作者将记录的机器人轨迹离线映射为 BFM 专属标签，不需要原 teleoperation 的身体动作命令。
- 在四个真实机器人任务的 SONIC→SONIC 设定中，GR00T N1.7 高层策略使用 TTCO 标签、原始命令标签、robot-state labels 的平均成功率分别为 **64.2% / 60.4% / 2.5%**。每任务、每标签来源 60 次 rollout。
- 逐任务结果并非一致增长；例如 Cup in basket 为 55.0% / 55.0%，Basket on shelf 为 50.0% / 53.3%（TTCO / 原始命令）。结论应限于该评测协议。

## 页面锚点

- #method — Test-Time Command Optimization
- #results — 三个 BFM 的 tracking fidelity
- #compute — 测试时计算扩展
- #demos — 高层策略与示范复用

## 关联资料

- [论文来源档案](../papers/ttco_test_time_compute_humanoid_bfm_arxiv_2610_14896.md)
- [官方代码仓库档案](../repos/bfm_ttc.md)
