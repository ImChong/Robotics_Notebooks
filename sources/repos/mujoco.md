# MuJoCo（google-deepmind/mujoco）

> 来源归档

- **标题：** MuJoCo
- **类型：** repo
- **来源：** Google DeepMind
- **链接：** https://github.com/google-deepmind/mujoco
- **官方文档：** https://mujoco.readthedocs.io/
- **PyPI：** https://pypi.org/project/mujoco/
- **npm（WASM）：** https://www.npmjs.com/package/@mujoco/mujoco
- **入库日期：** 2026-04-11（2026-06-30 增补 WASM / 生态）
- **一句话说明：** Google DeepMind 维护的开源通用物理引擎（Multi-Joint dynamics with Contact），强调刚体动力学、接触仿真、控制与优化友好性；主仓含 C API、Python 绑定、**官方 JavaScript/WASM 绑定**、MJX（JAX）与 Unity 插件。
- **沉淀到 wiki：** 是 → [`wiki/entities/mujoco.md`](../wiki/entities/mujoco.md)、[`wiki/entities/mujoco-wasm.md`](../wiki/entities/mujoco-wasm.md)

## 仓库结构（读者心智模型）

| 子树 / 产物 | 说明 |
|-------------|------|
| C 核心 + `simulate` | 预分配低层数据结构、XML 编译器、OpenGL 交互可视化 |
| `python/` | 官方 Python 绑定（`pip install mujoco`），附 Colab 教程（LQR、rollout、MJX 等） |
| `mjx/` | **MuJoCo XLA**：JAX 重实现，PyPI 包 `mujoco-mjx` |
| `wasm/` | **官方 JavaScript/TypeScript + WebAssembly 绑定**，npm 包 `@mujoco/mujoco`（含单线程默认版与 `/mt` 多线程版） |
| Unity 插件 | C# 绑定，见文档 unity 章节 |

## 官方 WASM 绑定要点（`wasm/README.md`）

- **安装**：`npm install @mujoco/mujoco`；ESM 包，需 bundler/dev server 正确提供 `.wasm` 资产。
- **单线程（默认）**：`import loadMujoco from '@mujoco/mujoco'`；兼容所有现代浏览器。
- **多线程（`/mt`）**：`import loadMujoco from '@mujoco/mujoco/mt'`；需 **Cross-Origin Isolation**（`COOP: same-origin` + `COEP: require-corp`）以启用 `SharedArrayBuffer`。
- **版本对齐**：npm 版本号跟随 MuJoCo 发行版（如 3.5.0 ↔ MuJoCo 3.5.0）。
- **成熟度**：README 标明 **WIP**；主 API（`mj_step`、`mj_loadXML` 等）已测，部分 API（如 `mjspec`）与 Windows 支持仍待完善；开发主要在 Linux + Chrome。
- **社区渊源**：官方 README 致谢社区项目 [stillonearth](https://github.com/stillonearth)、[zalo/mujoco_wasm](https://github.com/zalo/mujoco_wasm)；[mjswan](https://github.com/ttktjmt/mjswan) 在此基础上扩展实时策略控制与交互施力等。

## 发行与版本

- **预编译二进制**：GitHub Releases（Linux x86-64/AArch64、Windows x86-64、macOS universal）。
- **Python**：`pip install mujoco`（≥ Python 3.10），wheel 内含 MuJoCo 副本。
- **版本节奏**：目标每月首周发版；3.5.0 起采用修改版 Semantic Versioning（见 `VERSIONING.md`）。

## 为什么值得保留

- 机器人 RL、生物力学、轨迹优化与 Sim2Real 的**事实标准**物理后端之一。
- **浏览器 WASM 路径**使论文页、教学 demo、ONNX policy 在线 Sim2Sim 无需本地安装成为可能（与 [robot-viewer](robot-viewer.md)、[RL Sim2Sim Demo](../sites/rl-sim2sim-demo-website.md) 等生态互链）。
- 与 **MJX**、**dm_control**、**MuJoCo Playground** 等同仓或官方生态形成完整训练栈。

## 对 wiki 的映射

- [MuJoCo（实体页）](../../wiki/entities/mujoco.md)
- [MuJoCo WASM（浏览器绑定实体页）](../../wiki/entities/mujoco-wasm.md)
- [MuJoCo MJX](../../wiki/entities/mujoco-mjx.md)
- [MuJoCo Playground](../../wiki/entities/mujoco-playground.md)
- [dm_control](../../wiki/entities/dm-control.md)
