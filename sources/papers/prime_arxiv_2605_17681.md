# PRIME: Physically-consistent Robotic Inertial and Motion Estimation for Legged and Humanoid Robots（arXiv:2605.17681）

> 来源归档（ingest）

- **标题：** PRIME: Physically-consistent Robotic Inertial and Motion Estimation for Legged and Humanoid Robots
- **缩写 / 框架：** **PRIME**（Physically-consistent Robotic Inertial and Motion Estimation）
- **类型：** paper / humanoid / sysid / contact / map / rss
- **arXiv：** <https://arxiv.org/abs/2605.17681>（Submitted 2026-05-17；PDF：<https://arxiv.org/pdf/2605.17681>；HTML：<https://arxiv.org/html/2605.17681>）
- **项目页：** <https://jkangkjr.github.io/PRIME-project/> — 归档见 [`sources/sites/prime-project.md`](../sites/prime-project.md)
- **代码仓：** <https://github.com/well-robotics/PRIME> — 归档见 [`sources/repos/prime-system-id.md`](../repos/prime-system-id.md)
- **作者：** Jiarong Kang、Kunzhao Ren、Tao Pang、Xiaobin Xiong（通讯）
- **机构：** 威斯康星大学麦迪逊分校（University of Wisconsin–Madison）；通讯作者现单位上海创智学院（Shanghai Innovation Institute）
- **发表：** Robotics: Science and Systems (RSS) 2026
- **入库日期：** 2026-09-04
- **一句话说明：** 把接触丰富运动重建写成参数全信息 MAP：用运动学 + 执行器命令联合估计轨迹、摩擦接触力与物理一致惯量；可微 Anitescu 接触 + Crocoddyl/FDDP；G1 / Go2 验证。

## 开源状态（步骤 2.5）

- **项目页核查（2026-09-04）：** [PRIME-project](https://jkangkjr.github.io/PRIME-project/) 已上线，摘要与 G1/Go2 辨识表、测力台对比齐全。
- **仓库核查（2026-09-04）：** [well-robotics/PRIME](https://github.com/well-robotics/PRIME) 公开；README 给出 `g1_real_dance_1` 等 CMake 目标、XML 配置与输出 `inertia_identification.txt` / `f_rollout.csv`。依赖 Pinocchio 3.4 / Crocoddyl 栈。**BSD-3-Clause**。
- **结论：已开源**（实验可运行；权重/大规模数据集不随仓，日志 CSV 在各 `experiments/*/data/`）。

## 摘录 1：问题与主张

- **痛点：** EKF / 动捕只恢复运动学；接触力、接触时刻、惯量未观测 → 接触丰富段违反刚体动力学。
- **主张：** MAP 同时细化轨迹、摩擦接触与惯量；接触用平滑互补 + Anitescu 摩擦，避免硬互补导致的优化断裂。
- **求解：** Parameter Full-Information Estimation + FDDP（Crocoddyl）。论文称 1000 步视界在 i9-13900 上约 **200 s** 内收敛。
- **下游：** 校准惯量服务估计/控制；力–接触标注轨迹可喂模仿学习 / VLA / 基础模型。

**对 wiki 的映射：** 补强 [`wiki/entities/prime-system-id.md`](../../wiki/entities/prime-system-id.md)（既有 HMI 开源主表 draft，**不另造** `paper-prime`）；交叉 [闭环惯量标定](../../wiki/concepts/humanoid-closed-loop-inertia-calibration.md)。

## 摘录 2：机制

- 惯性参数用 Log-Cholesky / 伪惯量正定约束，保证物理可行。
- 接触隐式、无需额外力传感器即可重建接触力（G1 测力台对照）。
- 实现把可微接触 dynamcis 接到 Crocoddyl action / 参数增广状态；实验用 XML 指定 URDF、接触帧、辨识连杆与 CSV（q/v/u）。

**对 wiki 的映射：** 实体页画 MAP 数据流与源码运行时序图（对齐 README 实验入口）。

## 摘录 3：评测（项目页表 + 论文）

| 设定 | 要点 |
|------|------|
| Go2 仿真 +3 kg | 躯干质量 6.927 → 估 **9.975** kg（接近 +3） |
| Go2 仿真 \(c_z-0.1\) m | \(c_z\) −0.005 → **−0.105** m |
| Go2 真机腹板 +4.6 kg | 质量估 **11.740** kg |
| G1 真机总重 +2.91 kg | 躯干质量 9.60 → **13.02** kg |
| G1 测力台 | 带 ID：RMSE_F **24.486 N**；不带 26.141 N |

**对 wiki 的映射：** 强调「接触隐式 + 惯量联合」相对「先运动学再单独 SysID」；数字以项目页 / 论文表为准。

## 建议 wiki 动作

- 维护 **`wiki/entities/prime-system-id.md`**（从 HMI draft 升为 complete）、**`sources/repos/prime-system-id.md`**、**`sources/sites/prime-project.md`**。
- 交叉 SysID / 闭环惯量标定 / Crocoddyl / 接触估计；公众号读法见 [`wechat_humanoid_zhiyan_inertia_closedloop_calib_2026-08-26.md`](../blogs/wechat_humanoid_zhiyan_inertia_closedloop_calib_2026-08-26.md)。
