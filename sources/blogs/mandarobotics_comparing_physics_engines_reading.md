# Manda Robotics：机器人物理仿真引擎四栈基准与源码深挖通读

> 来源归档（原文通读 · 支撑 `wiki/concepts/robot-simulation-three-layers.md`）

- **标题：** Comparing physics engines for robotics simulation & Companion: An overview of physics simulators for robotics
- **作者 / 来源：** Manda Robotics（2026-09-30 / 2026-10-05）
- **类型：** blog / technical benchmark & source code audit
- **原文：**
  - 主评测：<https://mandarobotics.com/blog/comparing-physics-engines/index.html>
  - 伴随原理与源码深挖：<https://mandarobotics.com/blog/comparing-physics-engines/primer.html>
  - 运行时协议：<https://mandarobotics.com/blog/comparing-physics-engines/assets/runtime/RUNTIME-PROTOCOL-2026-09-25.md>
  - 碰撞链审计：<https://mandarobotics.com/blog/comparing-physics-engines/assets/collision-chain/audit.md>
- **入库日期：** 2026-11-28
- **一句话说明：** 通过同一 Franka Panda 资产与显式力矩控制器对照 PhysX（Isaac Sim 6.1）、Newton（MuJoCo-Warp 3.11.0）、MuJoCo CPU 3.11.0 与 Genesis 1.4.1 四套仿真栈，揭示「仿真框架 / 物理引擎 / 渲染器」三层解耦、软接触与硬接触源码实现差异、刚柔耦合陷阱及单环境 CPU vs GPU 批吞吐对比。
- **沉淀到 wiki：** 是 → [`wiki/concepts/robot-simulation-three-layers.md`](../../wiki/concepts/robot-simulation-three-layers.md)
- **通读：** 已完成
- **核对范围：** 通读主评测 `index.html`（§§1–8，929 行提取文本）、伴随长文 `primer.html`（§§1–5 含软/硬接触数学与源码摘录、三组耦合架构与移植清单，765 行）、`RUNTIME-PROTOCOL-2026-09-25.md` 与 `collision-chain/audit.md`。

## 核心摘录（均出自原文）

1. **三层解耦框架：** 伴随文 §1 明确区分 **Simulator / framework**（Isaac Lab、Genesis World、Gazebo、MuJoCo Playground —— 管资产导入 URDF/MJCF/USD/SDF、任务与控制逻辑）、**Physics engine**（PhysX、MuJoCo、Genesis、DART、Bullet —— 管碰撞检测、接触模型、约束求解、积分器）与 **Renderer**（独立出图；场景渲染逼真不代表物理准确）。
2. **四套对比配置与接触族分野：** MuJoCo CPU（float64）、Newton/MuJoCo-Warp（float32 GPU）与 Genesis（float64 GPU）同属 **广义坐标 + 凸优化软接触**；PhysX（Isaac Sim 6.1，CPU TGS，100 position substeps）属 **简约坐标 + 硬约束非穿透**。实测同一 Panda 粗任务成功率一致，但滑块接触力峰值差达 **10 倍**（1.06 N vs 10.46 N），跌落穿透在软接触下可达 **7.57–7.84 mm**（10 ms `solref` 弹簧近似使然）而 PhysX `< 0.001 mm`；紧密插孔（0.10 mm 间隙）中三套软接触栈经力反馈均达标而 PhysX 多次触发 40 N/4 Nm 过载停机。
3. **导入默认值陷阱与源码机理：** 初期 Genesis 单摆铰链 1 s 内偏离基准 $51.4^\circ$，根因是导入器注入 `0.1 kg·m²` 默认 `armature` 而非求解器公式错误；Genesis 1.4.1 刚柔耦合采用 Drake 风格 SAP 耦合器，默认 `coup_friction = 0.1` 与仅 `5` 次迭代导致等效摩擦跌至 $\sqrt{0.5\times 0.1}\approx 0.22$ 而抓不住软球；Newton 1.5.2 采用 MuJoCo-Warp + VBD **滞后一拍（lagged proxy）** 显式耦合，把 MJCF `solref=(0.01,1)` 转成 $k_d=200\ \mathrm{N\cdot s/m}$ 后违背稳定条件 $\alpha = hc/m < 2$ 触发 1.92 kN 发散。
4. **单环境与千级并行性能反转：** 在接触自由的 Panda 1 ms 测试（RTX PRO 5000 + Ryzen 9 9900X）中，单环境 CPU MuJoCo 达 **0.17M steps/s**（~167k steps/s），远快于单环境 GPU（MJWarp ~5.7k、Genesis ~1.9k）；但扩展至 **8,192 并行环境** 时，MuJoCo-Warp 达 **21.6M steps/s**、Genesis 达 **11.8M steps/s**（至 131,072 环境分别达 35.2M / 26.4M steps/s），决定「CPU 单场景调试 + GPU 批量训练」的工程分工（前提是导入质量/惯量/接触族逐项复核）。

## 对 wiki 的映射

- [`wiki/concepts/robot-simulation-three-layers.md`](../../wiki/concepts/robot-simulation-three-layers.md)
