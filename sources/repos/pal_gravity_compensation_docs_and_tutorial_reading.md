# PAL 重力补偿控制器：官方文档与教程仓通读

> 来源归档（原文通读 · 支撑 `wiki/concepts/gravity-compensation.md`）

- **标题：** PAL OS 25.01 *Gravity Compensation controller* 文档页 + `pal-robotics/gravity_compensation_controller_tutorial` 仓库
- **类型：** site + repo（官方文档 / 源码）
- **原文：** <https://docs.pal-robotics.com/25.01/hardware/controllers/gravity-compensation.html>（2026-11-07 抓取）；仓库 <https://github.com/pal-robotics/gravity_compensation_controller_tutorial>（`git clone --depth 1`，2026-11-07）
- **入库日期：** 2026-11-07
- **一句话说明：** TIAGo 系列的重力补偿 effort 控制器：文档讲用法与 YAML 参数；教程仓用 RBDL `InverseDynamics(q,0,0)` 算 g(q)，叠加粘滞 / 静摩擦并换算成电流指令。
- **核对范围：** 文档页全文（用法、YAML 参数、注意事项）；教程仓 `README.md`、`src/gravity_compensation_controller.cpp` 全文 280 行（`initRequest`、`init`、`update`、`starting`、`stopping`）、`config/` 与 `test/` 文件名。**未阅读**头文件、各 `config/*.yaml` 数值、测试代码和 CHANGELOG，未编译或运行；生产控制器 `pal_controllers/GravityCompensationController` 未开源，只有文档层面信息。
- **通读：** 已完成

## 核心摘录（均出自原文）

- **文档页：** 控制器使用"基于电流控制的 effort 接口"；激活后用户推动连杆即可改变手臂构型，用于 kinesthetic teaching（示教学习）；可经 Web Commander 或 `ros2 control switch_controllers --activate <arm>_gravity_compensation_controller --deactivate <arm>_controller` 切换，且 ros2_control 不允许两个控制器共用同一资源，必须同时停掉默认控制器；**仅用于 TIAGo PRO / TIAGo 的手臂，头与躯干不可用**。YAML 参数：`type: pal_controllers/GravityCompensationController`、`command_joints`（7 轴）、`root_link`（示例 `torso_lift_link`）、`tip_links`、`torque_gain`（作用于计算出的力矩的增益，示例 1.0）、每关节 `command_interface`（`.../effort`）、`static_friction`、`viscous_friction`（示例多为 0.05，关节 2 的静摩擦示例值为 −1.0）；提示：TIAGo 模型通常还要配 `motor_torque_constant` 与 `reduction_ratio`，SEA 臂的这两项已在底层控制里计入。
- **教程仓 README：** 示范如何用 ros_control 实现克服重力的 effort 控制器；运行前须停掉与之共用资源的控制器（如 `arm_controller`）；**启动文件不要关闭，否则手臂会掉落**，须先切回 `arm_controller` 再取消；launch 参数 `robot`、`end_effector`（pal-gripper / pal-hey4 / schunk-wsg）、`simulation`、`spawn`、`controller_ns`；换其他机器人 / 末端需重新生成配置文件。
- **源码 `update()` 流程：** 读各关节位置 → `RigidBodyDynamics::InverseDynamics(rbdl_model_, q, 0, 0, tau)` 得到保持当前位置的最小力矩（即 g(q)）→ 对每个 `ACTUATED` 关节加 `viscous_friction × 速度`，速度大于 `velocity_tolerance` 时加静摩擦、否则减静摩擦 → 除以 `motor_torque_constant × reduction_ratio` 得到电流 / effort 指令 → 非有限值则报错不下发；`ACTUATED_NO_CONTROL` 关节恒发 0；`stopping()` 把所有执行关节指令置零。`init()` 从参数服务器读 URDF，建 RBDL 固定基模型（可按 `robot_model_chains` 取子链），三类关节：actuated / no_control / static（仅读状态）；摩擦增益经 `ddynamic_reconfigure` 可在线调。

## 对 wiki 的映射

- [`wiki/concepts/gravity-compensation.md`](../../wiki/concepts/gravity-compensation.md)

## 参考来源

- 文档页与仓库如上；既有 ingest 归档 [`../sites/pal-robotics-gravity-compensation.md`](../sites/pal-robotics-gravity-compensation.md)、[`gravity-compensation-controller-tutorial.md`](gravity-compensation-controller-tutorial.md)
