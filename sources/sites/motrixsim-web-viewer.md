# MotrixSim Web Viewer（浏览器物理仿真）

> 来源归档

- **标题：** MotrixSim Web Viewer / Motrix Viewer
- **类型：** site（官方 Web 仿真入口 + 用户指南）
- **URL：**
  - 在线 Viewer：<https://motrix.motphys.com/>
  - 用户指南：<https://motrixsim.readthedocs.io/en/latest/user_guide/getting_started/motrixsim_web.html>
- **入库日期：** 2026-06-18
- **一句话说明：** 基于 **WebAssembly** 的 MotrixSim 浏览器端物理仿真 Viewer：拖入完整模型文件夹即可加载 MJCF / URDF / JSON 场景，内置 Online 示例与 Customize 本地会话资源，支持播放/步进/重置/重载与物理拖拽交互。
- **沉淀到 wiki：** [Motrix](../../wiki/entities/motrix.md)

---

## 页面能力要点（策展）

### 入口与界面

- 现代浏览器 + **WebAssembly** 支持即可打开；无需本地安装桌面客户端。
- 典型布局：**左侧文件树**（`Online` / `Customize`）、**中央 3D 视口**、**顶部工具栏**（播放与场景控制）。

### 模型加载（Web 平台关键约束）

- Web 端**不能像桌面版那样直接访问本地文件系统**；推荐 workflow 是**将整个模型文件夹拖入浏览器窗口**。
- 推荐目录结构示例：

```text
boston_dynamics_spot/
├── scene.xml
├── meshes/
├── textures/
└── ...
```

- 依赖资源（mesh、texture 等）需与场景文件保持**相对路径**；因此应拖**整个文件夹**，而非仅拖单个 `.xml` / `.urdf` / `.json`。
- 加载后文件出现在左侧 `Customize`；在文件树中**点击场景文件**即可 spawn 场景。

### `Online` vs `Customize`

| 来源 | 性质 | 适用 |
|------|------|------|
| **Online** | 站点部署的只读内置示例；加载 online manifest 后自动出现 | 快速体验官方 demo、共享示例资产 |
| **Customize** | 用户拖入的文件夹/文件；**仅当前浏览器会话**有效 | 自测 MJCF、URDF、MSD、mesh、texture |

### 工具栏与快捷键

**播放控制：** Play / Pause / Next（单帧步进）

**场景控制：** Reset（回到初始状态）、Reload（重载当前模型及资产）

**常见输入：**

| 输入 | 作用 |
|------|------|
| 左键拖拽 | 轨道相机 |
| 滚轮 | 缩放 |
| 中键拖拽 / 触控板平移手势 | 平移相机 |
| Space | 暂停 / 继续 |
| F10 | 单步前进 |
| Ctrl+E | 重置场景 |
| Ctrl+R | 重载模型 |
| F11 | 全屏（若浏览器支持） |
| Ctrl + 左键拖拽物体 | 物理拖拽（需在 app 配置中启用 physics drag） |

### 典型 workflow

1. 打开 <https://motrix.motphys.com/>
2. 拖入完整模型文件夹
3. 在 `Customize` 中点击场景文件
4. 鼠标检视场景；工具栏播放 / 暂停 / 重置 / 重载

### 与 Motrix 生态的关系

- **MotrixSim 引擎**的零安装、可分享演示形态；适合快速验模、教学与对外展示。
- [UniLab 项目页](unilabsim-project.md) 等叙事中提到的「浏览器 MotrixSim 策略试玩」与此 Viewer 同属 Web 侧 MotrixSim 能力。
- 重度 RL 训练与批量并行仍走 **MotrixLab / 桌面 MotrixSim**；Web Viewer 侧重**交互式检视与轻量仿真**，非训练栈替代品。

## 对 wiki 的映射

- [Motrix（实体页）](../../wiki/entities/motrix.md) — 补充 MotrixSim Web Viewer 小节与参考来源
- [UniLab](../../wiki/entities/unilab.md) — 浏览器 demo 语境可交叉引用 Motrix 页
