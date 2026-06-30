# mujoco_wasm（zalo/mujoco_wasm）

> 来源归档

- **标题：** mujoco_wasm
- **类型：** repo
- **作者：** zalo
- **链接：** https://github.com/zalo/mujoco_wasm
- **在线 Demo：** https://zalo.github.io/mujoco_wasm/
- **许可：** MIT
- **入库日期：** 2026-06-30
- **一句话说明：** 基于 **DeepMind 官方 MuJoCo WASM 绑定** 的浏览器端 MuJoCo 演示套件（Three.js 渲染 + `examples/`）；项目早期曾自行编译 WASM，现转为官方 `@mujoco/mujoco` 之上的轻量示例与教学入口。
- **沉淀到 wiki：** 是 → [`wiki/entities/mujoco-wasm.md`](../wiki/entities/mujoco-wasm.md)

## 项目定位演变

README 明确说明：

> This project used to be a WASM compilation and set of javascript bindings for MuJoCo, but since Deepmind completed the official MuJoCo bindings, this project is now just a small demo suite in the `examples` folder.

因此入库价值在于：

1. **历史与选型**：理解「社区先行 WASM → 官方 `@mujoco/mujoco` 接管」的迁移路径。
2. **最小可运行范例**：`npm install` 后通过 HTTP server 即可加载 MJCF、步进 `mj_step`、访问 `qpos`/`qvel`/`ctrl` 等 TypedArray 缓冲区。
3. **生态锚点**：被 [google-deepmind/mujoco](mujoco.md) 官方 README 列为 JavaScript 绑定的社区渊源之一；下游项目（如 [robot-viewer](robot-viewer.md)、[RoboPianist](https://kzakka.com/robopianist/)）常以此为参考实现浏览器内 MuJoCo。

## 技术栈与 API 形态

- **依赖**：`npm install` 拉取 Three.js 与 **MuJoCo 官方 WASM 绑定**（非自编译 fork）。
- **典型流程**（摘自 README）：
  - `import load_mujoco from "./dist/mujoco_wasm.js"`
  - Emscripten 虚拟文件系统：`FS.mkdir` / `FS.mount` / `FS.writeFile` 加载 MJCF
  - `MjModel.loadFromXML` → `MjData` → `mj_step` / `mj_forward` / `mj_resetData` / `mj_applyFT`
- **当前跟踪版本**：README 标明 MuJoCo **3.3.8**（随上游绑定更新）。

## 与官方绑定的关系

| 维度 | zalo/mujoco_wasm | google-deepmind/mujoco `wasm/` |
|------|------------------|--------------------------------|
| 维护方 | 社区（zalo） | Google DeepMind（canonical） |
| npm 包 | 演示仓内 `dist/` | `@mujoco/mujoco` |
| 多线程 `/mt` | 未强调 | 官方提供，需 COOP/COEP |
| 适用场景 | 快速 demo、教学、fork 起点 | 生产集成、版本对齐、长期维护 |

**新工程应优先 `@mujoco/mujoco`**；本仓适合作为「浏览器里如何挂 MJCF + 步进仿真」的参考实现。

## 对 wiki 的映射

- [MuJoCo WASM（实体页）](../../wiki/entities/mujoco-wasm.md)
- [Robot Viewer](../../wiki/entities/robot-viewer.md) — 消费 `mujoco_wasm` 的多格式 Web 查看器
- [MuJoCo（主引擎）](../../wiki/entities/mujoco.md)
