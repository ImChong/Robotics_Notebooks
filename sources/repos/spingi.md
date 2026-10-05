# Spingi（ceccode/spingi）

> 来源归档（开源仓库）

- **标题：** Spingi — Sim-first Physical Agent Runtime for humanoid robots
- **类型：** repo / humanoid robotics / task-level agent runtime
- **代码：** <https://github.com/ceccode/spingi>
- **项目页 / Viewer：** <https://spingi-viewer.netlify.app/>
- **许可：** 仓库 Apache-2.0；仓内 Unitree G1 MuJoCo 模型按其 BSD-3-Clause 来源许可保留 notices
- **语言与环境：** Python 3.11+、uv；Web Viewer 使用 TypeScript / Three.js、Node 20+
- **状态（README，2026-10-04）：** runtime M3 complete；viewer v0.2
- **入库日期：** 2026-10-05
- **一句话说明：** 用可校验的技能计划驱动 MuJoCo 中的 Unitree G1 原型，集中演示 LLM 任务规划、执行器、安全监控、episode 记录与浏览器回放；当前不是 G1 真机控制栈。

---

## 组成与复现入口

| 目录 / 组件 | 职责 |
|-------------|------|
| `runtime/` | Python Physical Agent Runtime；plan、skills、executor、adapter、perception、safety 与 episode writer |
| `runtime/sim/` | MuJoCo G1 模型与 YAML 场景；模拟碰撞、移动、相机与对象状态 |
| `viewer/` | 静态 Three.js Viewer；读取 episode ZIP 并回放机器人、对象、事件与相机帧 |
| `docs/episode-format.md` | Runtime 与 Viewer 间的 episode 格式契约 |
| `adr/` | 记录架构决策，包括 LLM 边界、运动学模拟器与急停语义 |

从仓库根目录进入 runtime 后，README 给出的最短仿真路径：

```bash
cd runtime
make setup
make demo-sim
```

运行物流样例并导出可分享 episode：

```bash
uv run spingi run plans/demo_material_runner.yaml --scene sim/scenes/warehouse_small.yaml --adapter sim --record --zip
```

在浏览器打开 Viewer，加载生成的 ZIP 或选择内置样例。Viewer 在浏览器本地解析 episode；README 说明文件不会上传。

## 运行时边界

- LLM 只生成由技能白名单约束的结构化任务计划，不能直接发送关节位置、速度或力矩；计划通过同一套校验与 Executor。
- 当前技能包括 navigate、detect、pick、place、inspect、wait_for_human、say；计划是有序步骤，无分支/循环，异常按设定策略重试、跳过、终止或请求人工。
- `SimAdapter` 使用 MuJoCo 的 G1 模型做**运动学底盘位移**：躯干/骨盆按目标平移，双腿保持站立姿态；MuJoCo 物理主要用于障碍碰撞检测，手臂未实现实际抓取运动。
- Runtime 含有 geofence、限速、电量、robot-time deadline、watchdog 与 terminal e-stop 的软件监控；这些机制仍需真实硬件 adapter 与硬件侧超时配合，不能视为已完成安全认证。
- 当前没有可用 G1 真机 adapter；README 将真机 adapter 标为后续 M4。MuJoCo 中的成功率门槛是软件仿真回归，不代表 sim-to-real 已验证。

## Episode 与数据导出

每次 `spingi run` 写出 plan、scene、events.jsonl、trajectory.jsonl、manifest 与可选相机帧/视频，ZIP 可用于独立 Viewer 回放。Runtime 提供 LeRobotDataset v3 导出，但当前只写 10 Hz 的低维 base x/y/yaw 与 gripper state/action；相机帧并非固定频率，暂未导出为连续图像序列。

## 对 wiki 的映射

- [Spingi 实体页](../../wiki/entities/spingi.md)
- [Spingi Viewer 项目页归档](../sites/spingi-viewer.md)
- [Unitree G1 实体页](../../wiki/entities/unitree-g1.md)
