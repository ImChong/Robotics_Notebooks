# project_superdex

> 来源归档

- **标题：** Project SuperDex
- **类型：** repo
- **机构：** Meta / facebookresearch
- **链接：** https://github.com/facebookresearch/project_superdex
- **主页：** https://projectsuperdex.com/
- **Stars：** ~611（2026-09-09）
- **入库日期：** 2026-09-09
- **一句话说明：** Meta 开源的灵巧操作统一仿真平台：自研接触优先物理引擎 + 机器人 SDK + Studio 资产工具 + Gymnasium 风格 Lab，打通场景搭建、仿真、遥操作（计划）到 RL 策略训练。
- **代码：** https://github.com/facebookresearch/project_superdex（**已开源**，Apache 2.0）
- **沉淀到 wiki：** 是 → [`wiki/entities/project-superdex.md`](../../wiki/entities/project-superdex.md)

---

## 四核心模块（README）

| 模块 | 职责 |
|------|------|
| **SuperDex Physics** | 接触优先（contact-first）物理引擎，触觉/接触密集型操作仿真底座 |
| **SuperDex Robotics** | 机器人 SDK：定义与组合、控制器、传感器、执行器 → 完整仿真配置 |
| **SuperDex Studio** | 桌面 GUI：CAD/机器人描述 → 原生资产（bot、mesh、task prefab、scene）编辑与验证 |
| **SuperDex Lab** | Gymnasium 风格 API，连接仿真与策略开发（RL / MPC / system-ID）；**early preview** |

计划能力：**SuperDex Teleop**（Q4 2026）— Quest 3 端侧、手追踪+控制器混合、UE5 虚拟遥操作。

---

## 平台与依赖

- **OS：** Linux x86_64、Windows x86_64、macOS ARM
- **Python：** 3.12（预编译 wheel 仅 3.12；未来 abi3）
- **快速安装：** `uv venv` → `uv pip install superdex`
- **源码构建：** `uv sync --extra gui`（含 Studio / Physics Debugger / mesh-cli）；`--extra core` 含 physics + robotics + lab
- **编译器：** Clang 17+（Linux/macOS）；MSVC + ClangCL（Windows）；GCC/MSVC 非官方 CI 支持
- **精度：** 默认 float32；`SUPERDEX_PRECISION=double` 或 `--extra double` 启用 fp64

---

## 示例入口（README）

```bash
# Physics
uv run --no-project superdex_physics/examples/example_tendon_comparison.py

# Robotics（OSC + JSC 控制示例）
uv run --no-project superdex_robotics/examples/control/example_osc_jsc_control.py

# Studio GUI
uv run --no-project superdex-studio
```

> 仓内 `uv run` 须加 `--no-project`，否则会触发源码构建。

---

## 文档

- Physics：https://projectsuperdex.com/physics/docs/overview/
- Robotics：https://projectsuperdex.com/robotics/docs/overview/
- Studio：https://projectsuperdex.com/studio/docs/overview/
- Lab：https://projectsuperdex.com/lab/docs/overview/

---

## 开源核查（步骤 2.5）

| 项 | 结论 |
|----|------|
| GitHub 仓 | **已开源**（Apache 2.0 + CC BY 4.0 资产） |
| PyPI | `superdex` 包可 `pip install` |
| 可运行 | Physics/Robotics 示例 + Studio GUI（Linux 需 X11 + OpenGL 4.1） |
| Lab | early preview，README 称将大幅改进 |
| Teleop | **未发布**（Q4 2026 路线图） |
| 论文 | 暂无正式 citation 块 |

---

## 与生态关系

- **MuJoCo / Isaac Lab：** 通用 RL 仿真栈；SuperDex 自研物理引擎，主攻 **接触/触觉** 灵巧操作而非 GPU 大规模并行
- **DexBench：** 工业真机 OSC 规格；SuperDex 是 **仿真+authoring+RL 平台**，非同一评测榜
- **RoboCasa / ManiSkill：** 厨房/机械臂泛化 benchmark；SuperDex 强调 **多指灵巧与接触物理**

---

## 对 wiki 的映射

- [Project SuperDex](../../wiki/entities/project-superdex.md)
- [Contact-Rich Manipulation](../../wiki/concepts/contact-rich-manipulation.md)
- [DexBench](../../wiki/entities/dexbench.md)
