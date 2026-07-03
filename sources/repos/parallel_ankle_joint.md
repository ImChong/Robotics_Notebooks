# Parallel_Ankle_Joint

> 来源归档

- **标题：** Parallel_Ankle_Joint
- **类型：** repo
- **作者：** feidedao（微信 feidedaoRobot）
- **链接：** https://github.com/feidedao/Parallel_Ankle_Joint
- **Stars：** ~31（2026-07）
- **License：** MIT
- **语言：** Python
- **入库日期：** 2026-07-03
- **一句话说明：** 面向 **Unitree G1** 与 **天工（TienKung）** 并联踝的闭链 **IK / FK / 雅可比** 解析实现，附 MuJoCo 闭链仿真与解析解对比脚本。
- **沉淀到 wiki：** 是 → [`wiki/concepts/humanoid-parallel-joint-kinematics.md`](../../wiki/concepts/humanoid-parallel-joint-kinematics.md)

---

## 为什么值得保留

- 把教材/论文里的 **并联踝机构学** 落到 **两款量产/开源人形** 的可运行 Python 代码上，覆盖本库 [`humanoid-parallel-joint-kinematics`](../../wiki/concepts/humanoid-parallel-joint-kinematics.md) 概念页第 1 层（几何运动学）与第 2 层（\(J_c\) 力/速度映射）的工程对照。
- 作者明确标注 **URDF/MJCF 为自行拼装、动力学参数不可信**，与 wiki 中「串联等效接口 ≠ 机构电机角」的误区警示一致，适合作为 **Sim2Real 前校验闭链解算** 的参考起点，而非直接当官方标定。

---

## 仓库结构

| 目录 / 文件 | 说明 |
|-------------|------|
| `2_g1_ankle/parallelJoint_g1.py` | G1 踝 **解析 IK**、迭代 **FK**、闭链 **雅可比** \(J_c\)（`ik` / `fw` / `Jac`）；含杆长几何常量与 pitch/roll 工作空间 |
| `2_g1_ankle/close_mujoco_g1.py` | MuJoCo 闭链踝仿真 |
| `2_g1_ankle/compare_g1.py` | 解析解与 MuJoCo 轨迹对比 |
| `2_g1_ankle/urdf/`、`meshes/` | 非官方自组装的踝部 URDF 与网格 |
| `5_tiangong/parallelJoint_tg10.py` | 天工踝同类 API（`parallelJoint_tg10` 类） |
| `5_tiangong/close_mujoco_tg10.py`、`compare_tg.py` | 天工 MuJoCo 与对比脚本 |

配套 B 站讲解：<https://www.bilibili.com/video/BV1og9VBhEPn/>

---

## 技术要点（摘录）

### 坐标与变量约定（G1 左腿示例）

- 两路电机绕 **x 轴**（roll 方向）转动；**y** 为右腿指向左腿（pitch 方向）；**pitch / roll = 0** 时定义为电机零点。
- 平台姿态用 **Ry(pitch) @ Rx(roll)** 组合；IK 在固定杆长约束下对两路 **theta1/theta2** 求解析解（`arcsin` 分支，带工作空间越界 `error_state`）。
- **雅可比**：由 \(J_\theta\)、\(J_x\) 与选择矩阵 \(G\) 得 \(J_c = J_\theta^{-1} J_x G\)，满足 \(\Delta\theta = J_c \Delta[\text{roll}, \text{pitch}]\)；力映射用 \(J_c^\top\)（见 `forceVelJac` 注释）。

### 与文献的关系

代码注释引用 *On the Comprehensive Kinematics Analysis of a Humanoid Parallel Ankle Mechanism*（与本库 [`humanoid_parallel_ankle_kinematics_ingest.md`](../papers/humanoid_parallel_ankle_kinematics_ingest.md) 第 1 条 ResearchGate 索引一致）。

### 使用注意（作者自述）

- 提供的 XML **非官方**，根据开源文件自行组装；**运动学几何应接近真机，动力学参数完全错误**。
- FK 为 **牛顿式迭代**（`fw`），依赖初值与步长 `dt`；工作空间外 IK 会打印 warning 并置 `error_state`。

---

## 对 wiki 的映射

| 主题 | 目标页面 |
|------|----------|
| 并联踝解算分层、\(J_A\) / 闭链雅可比 | [`wiki/concepts/humanoid-parallel-joint-kinematics.md`](../../wiki/concepts/humanoid-parallel-joint-kinematics.md) |
| G1 平台与 RL/部署栈 | [`wiki/entities/unitree-g1.md`](../../wiki/entities/unitree-g1.md) |
| 天工开源人形 | [`wiki/entities/tienkung-humanoid-open-source.md`](../../wiki/entities/tienkung-humanoid-open-source.md) |
| 闭链 IK 通用参考 | 概念页「推荐继续阅读」中的 [closed-chain-ik-js](https://github.com/gkjohnson/closed-chain-ik-js) |
