# ttktjmt/mjswan 官方 README 与前后端运行时源码通读

> 来源归档（原文通读 · 支撑 `wiki/entities/mjswan.md`）

- **标题：** ttktjmt/mjswan（README、Python `Builder` 打包管线与浏览器 TypeScript `ControlStep` 运行时源码）
- **作者 / 来源：** ttktjmt（Apache-2.0；PyPI `mjswan` / npm `mjswan`）
- **类型：** repo README & Python/TypeScript source code
- **链接：** <https://github.com/ttktjmt/mjswan>
- **入库日期：** 2026-11-28
- **一句话说明：** 核读 `ttktjmt/mjswan` 官方 `README.md`、`src/mjswan/builder.py`、`src/mjswan/build/pipeline.py` 及前端引擎 `runtime.ts`、`controlStep.ts` 源码，验证其「Python 构建期打包（场景 `scene.mjz`/`scene.mjb` + MDP/策略 ONNX + `manifest.json` + 静态 SPA）→ 浏览器端帧预算控制环（`ControlStep.spend` + `mujoco` WASM `mj_step` + `onnxruntime-web` 推理 + Three.js/WebXR 交互）」双阶段架构。
- **开源状态：** 已开源，可运行（`pip install mjswan` / `npm install mjswan`，Apache-2.0，商用或企业页按 README License 保留面板天鹅图标）
- **沉淀到 wiki：** 是 → [`wiki/entities/mjswan.md`](../../wiki/entities/mjswan.md)
- **通读：** 已完成
- **核对范围：** 通读官方 `README.md`（102 行全文）及 `src/mjswan/builder.py`、`src/mjswan/build/pipeline.py`（写入结构与 `write_app`）、`src/mjswan/template/src/core/engine/controlStep.ts`、`src/mjswan/template/src/core/engine/runtime.ts`（核心类型与控制环结构）（2026-11-28 抓取 `main` 分支）。

## 核心摘录（均出自官方 README 与源码）

1. **构建期离线导出流（`Builder` → `build/pipeline.py:write_app`）：**
   - Python API 提供 `mjswan.Builder(base_path=..., mt=..., license=...)` 与 `Builder.from_mjlab(task_id, run_path=..., hf_repo_id=...)` 便捷工厂，可从 `mjlab` 任务 + W&B/HF checkpoint 导出 ONNX 并组装项目与场景；
   - `write_app()` 编译/拷贝前端 SPA 到输出目录（若存在自定义 MDP JS 项则额外编译 `assets/plugins.js`；若 `mt=True` 则写入 `_headers` 与 Service Worker 以满足多线程 WASM 的 COOP/COEP 要求），按 `<project-id>/<scene-id>/` 结构输出 `scene.mjz`/`scene.mjb`、`mdp/<mdp-id>/{obs,term,command,event,action}/*.onnx`、`policy/*.onnx`、`assets/`（motion/splat/skin）及全局 `manifest.json`。
2. **浏览器运行时帧预算闭环（`core/engine/controlStep.ts` + `runtime.ts`）：**
   - 对齐 `mjlab` viewer 的帧时间片调度模型：每帧渲染周期调用 `ControlStep.spend(budget, stepMs, deadline)`，避免控制与物理步滞后造成浏览器卡死：
     - `substep === 0` 时触发 `hooks.begin()`：从 `MjData` 构造观测（`PolicyStateBuilder` / `Observation` + 相机/接触/射线传感器）→ 调用 `onnxruntime-web/wasm` 推理策略并由 `ActionManager` / `OnnxActionProcess` 输出作动器目标；
     - 调用 `hooks.physics(from, to)` 执行 `stepPhysics`（`mj_step`）与交互施力/抓取注入（`InteractionManager` / `WeldHold` / WebXR `HandMocap`）；
     - 完成 `decimation` 步或遇到发散复位后执行 `hooks.end(diverged)`（评估 `TerminationManager`、`CommandManager`、`EventManager`），最后刷新 Three.js 场景与 `snapshot()`。
3. **纯客户端与许可注意点：** 全部仿真、ONNX 推理与 WebXR 手部交互在浏览器本地完成；PyPI wheel 不打包 `examples/`，本地跑内置示例需克隆仓库后用 `pip install -e '.[demo]'` 并运行 `mjswan demo`。

## 对 wiki 的映射

- [`wiki/entities/mjswan.md`](../../wiki/entities/mjswan.md)
