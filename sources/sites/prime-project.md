# PRIME 项目页（归档）

- **标题：** PRIME: Physically-consistent Robotic Inertial and Motion Estimation
- **类型：** site / project-page
- **URL：** <https://jkangkjr.github.io/PRIME-project/>
- **arXiv：** <https://arxiv.org/abs/2605.17681>
- **HTML：** <https://arxiv.org/html/2605.17681>
- **代码仓：** <https://github.com/well-robotics/PRIME>
- **入库日期：** 2026-09-04
- **配套论文：** [PRIME（arXiv:2605.17681）](../papers/prime_arxiv_2605_17681.md)
- **配套仓库：** [prime-system-id.md](../repos/prime-system-id.md)

## 一句话摘要

威斯康星大学麦迪逊分校 / 上海创智学院 的 **物理一致惯性 + 运动估计** 官方页：MAP 框架里联合细化运动学轨迹、摩擦接触力与惯量参数；Go2 / Unitree G1 真机与仿真对照，含躯干惯量辨识表与测力台 RMSE。

## 开源状态（步骤 2.5，2026-09-04 核查）

- 项目页链 arXiv 与方法摘要；代码入口为 [well-robotics/PRIME](https://github.com/well-robotics/PRIME)（RSS 2026 标注）。
- 仓内有 `include/crocoddyl/contact_id/`、`experiments/G1_real_dance_*`、`experiments/Go2_*`，README 给出 CMake 目标与 XML 配置运行步骤。
- **许可证：** BSD-3-Clause（保留上游 Crocoddyl 归属）。
- **结论：已开源**（可运行实验入口，非占位 README）。

## 公开信息要点（项目页）

- **作者：** Jiarong Kang、Kunzhao Ren、Tao Pang、Xiaobin Xiong（通讯；现单位上海创智学院）。
- **会议：** Robotics: Science and Systems (RSS) 2026。
- **主张：** 纯运动学重建在接触丰富运动中常违反刚体动力学；把摩擦接触与惯量当 MAP 变量，用可微 Anitescu 接触 + 平滑互补约束。
- **Go2 躯干辨识（项目页表）：** 原质量 6.927 kg；仿真 +3 kg → 估 9.975 kg；真机腹板 +4.6 kg → 估 11.740 kg。
- **G1 躯干辨识：** 原质量 9.60 kg；真机总重 +2.91 kg → 估 13.02 kg。
- **G1 测力台：** 带辨识 RMSE_F 24.486 N，不带 26.141 N；代价 1.016×10³ vs 1.880×10³。

## 对 wiki 的映射

- 升格 / 补强：[PRIME](../../wiki/entities/prime-system-id.md)
- 交叉：[人形整机闭环惯量标定](../../wiki/concepts/humanoid-closed-loop-inertia-calibration.md)、[SysID](../../wiki/concepts/system-identification.md)、[Crocoddyl](../../wiki/entities/crocoddyl.md)
