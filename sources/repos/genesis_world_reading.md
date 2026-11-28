# Genesis World 官方仓库 README 与内核步进源码通读

> 来源归档（原文通读 · 支撑 `wiki/entities/genesis-sim.md`）

- **标题：** Genesis-Embodied-AI/genesis-world（README 与 `Scene` / `Simulator` 核心步进源码）
- **作者 / 来源：** Genesis AI & Genesis-Embodied-AI contributors（原仓库名 `Genesis-Embodied-AI/Genesis` 已重命名为 `Genesis-Embodied-AI/genesis-world`）
- **类型：** repo README & Python source code
- **链接：** <https://github.com/Genesis-Embodied-AI/genesis-world>
- **官方文档：** <https://genesis-world.readthedocs.io/en/latest/>
- **入库日期：** 2026-11-28
- **一句话说明：** 核读 `Genesis-Embodied-AI/genesis-world` 官方 `README.md` 与 `genesis/engine/scene.py`、`genesis/engine/simulator.py` 核心源码，确认其「Simulation Interface → 统一多物理 Physics（Rigid / FEM / MPM / SPH / PBD / libuipc + SAP/Explicit Coupler）→ Render（Nyx / Luisa / Pyrender）→ Compiler（Quadrants，Taichi fork）」四层栈、双分支 `step()` 与可微 `backward()` 回放内核架构及 Apache-2.0 开源边界；并勘误既有页面中错误的 `arXiv:2412.12919`（实际为无关医学论文）与失效域名 `genesis-world.github.io`。
- **开源状态：** 已开源，可运行（`pip install genesis-world`，要求 `Python >= 3.12, < 3.15`，Apache-2.0）
- **沉淀到 wiki：** 是 → [`wiki/entities/genesis-sim.md`](../../wiki/entities/genesis-sim.md)
- **通读：** 已完成
- **核对范围：** 通读官方 `README.md`（247 行全文：四层架构、Catalog 示例、安装、`quadrants` / `genesis-nyx` 可选依赖与 BibTeX Citation）及 `genesis/engine/scene.py`（`build`、`step`、`backward`、`reset`）、`genesis/engine/simulator.py`（`step`、`_step`、`_reverse`、`substep`、`substep_pre_coupling`、`substep_post_coupling`）（2026-11-28 抓取 `main` 分支）。

## 核心摘录（均出自官方 README 与源码）

1. **四层架构与跨后端编译（README）：**
   - **Simulation Interface：** 支持 URDF / MJCF / OBJ / GLB / USD 资产解析、并行与异构环境、控制器、传感器（深度、IMU、LiDAR、触觉、接触力、表面距离）及内置 GUI；
   - **Physics：** 统一场求解器（Rigid / FEM / MPM / SPH / PBD / `libuipc` 选装），通过 Explicit Coupler 或 SAP 耦合不同物质；
   - **Render：** 内置面向机器人的实时渲染器 **Nyx**（`gs-nyx`）、Luisa 光追与 Pyrender 光栅；
   - **Compiler：** 基于 2025-06 自 Taichi fork 演进的 **Quadrants** 编译器，将 Python kernel 下沉至 CUDA / ROCm / Metal / Vulkan / x86 / ARM64，承载自动微分与 CUDA/GPU graphs。
2. **`Scene.build()` 与双分支 `Simulator.step()`（源码核对）：**
   - `Scene.build(n_envs, env_spacing, ...)` 生成并行环境网格偏移，调用 `self._sim.build(n_envs)` 后先预跑一次 `self._sim.step()` 触发 kernel 编译，随即 reset 回初态并初始化 `_visualizer` 与 `_recorder_manager`；
   - `Scene.step()` 执行 `_pre_step_callbacks`（可由交互视图 veto）与 `_recorder_manager.step()` 后进入 `Simulator.step()`：
     - 若处于 **纯刚体且无需梯度**（`_rigid_only and not _requires_grad`）快速路径，直接以 `untracked_launches()` 循环 `_substeps` 次 `rigid_solver.substep()` 并清零外力；
     - 否则进入 `_step()`（或在 `RolloutGraph.is_recording` 时记录 `StepOperation`），每个 substep 执行 `_coupler.preprocess()` → `substep_pre_coupling()` → `_coupler.couple()` → `substep_post_coupling()` → 活跃求解器 `changelog.commit()`；
     - 随后在 `torch.no_grad()` 下更新 `_sensor_manager.step()` 与渲染录制。
   - **可微反向传播（`Scene.backward(loss)`）：** 先快照当前 `SimState` 与 `steps`，执行 `torch.autograd.backward(loss)`，通过 `Simulator._reverse()` 按步从边界 checkpoint 正向重放至各 substep 并用 `LaunchRecorder.reverse()` 累积梯度，反向完成后恢复原前向物理快照以保留继续推演或 `reset()` 语义。
3. **链接勘误记录：** 既有一手未核页中的 `https://arxiv.org/abs/2412.12919` 实际指向医学血管 4DRGS 论文，`https://genesis-world.github.io/` 返回 404；Genesis README 的正式 BibTeX 引用指向项目 GitHub 仓与 2026 年 5 月 Genesis AI 技术博客，文档站为 <https://genesis-world.readthedocs.io/en/latest/>。

## 对 wiki 的映射

- [`wiki/entities/genesis-sim.md`](../../wiki/entities/genesis-sim.md)
