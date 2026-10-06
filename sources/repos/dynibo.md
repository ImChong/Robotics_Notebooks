# Dynibo

> 来源归档

- **标题：** dynibo — robot kinematics and dynamics library
- **类型：** repo
- **作者：** Xiaojie Xue（[xiaojie-xue](https://github.com/xiaojie-xue)）
- **仓库：** https://github.com/xiaojie-xue/dynibo
- **官方文档：** https://dynibo.readthedocs.io/（含中文用户指南）
- **Rust crate：** https://crates.io/crates/dynibo（Cargo.toml 当前 workspace 版本 0.5.1）
- **Python 包：** https://pypi.org/project/dynibo/（PyPI 页面当前展示 0.1.0；上传于 2026-08-05）
- **最新 GitHub release：** [v0.5.1](https://github.com/xiaojie-xue/dynibo/releases/tag/v0.5.1)，2026-09-15
- **Stars：** 43（GitHub 页面于 2026-10-06 核查；会变化）
- **许可证：** MIT（仓库代码；示例机器人描述各自保留第三方许可）
- **源码开放：** **已开源** — Rust 核心、Python / C / C++ 绑定、示例、基准与测试均在公开仓库。Release 含 Linux x86-64、macOS ARM64 和 Windows x86-64 的 C/C++ 预编译包；PyPI wheel 的版本显示为 0.1.0，与 GitHub v0.5.1 不同。
- **数据/权重：** 不适用；仓库包含用于示例和测试的机器人 URDF，不是训练数据集或模型权重。
- **一句话说明：** 以 Rust 实现运行时树状 URDF 运动学与动力学计算，支持固定/浮动基座及多语言接口；覆盖 FK、Jacobian、固定基 DLS-IK、质量矩阵、重力、RNEA 和 ABA。
- **沉淀到 wiki：** 是 → [wiki/entities/dynibo.md](../../wiki/entities/dynibo.md)（沿用已有实体，更新过时接口、版本和性能说明）

---

## 核心定位

Dynibo 不是运动规划或机器人控制框架，而是可以嵌入控制器和上层工具的运动学 / 刚体动力学库。它运行时从树状 URDF 建模，并在统一 Rust 核心上提供多语言 API。官方文档建议通过固定基 `Robot` 或浮动基 `FloatingRobot` 建立模型；浮动基计算显式接收 `BaseState`。

## 算法与输入边界（官方用户指南）

| 接口 | 作用 / 约定 |
|------|-------------|
| `forward_kinematics` | 关节配置到目标 link 世界坐标系位姿 |
| `jacobian` / `jacobian_derivative` | 几何 Jacobian 与时间导数；浮动基前六列对应基座运动 |
| `forward_velocity_kinematics` / `forward_acceleration_kinematics` | 目标 link 或 tool 点的速度 / 加速度 |
| `inverse_kinematics` | DLS 数值 IK；仅固定基 `Robot` 提供 |
| `mass_matrix` | 对称广义惯性矩阵 |
| `velocity_product_forces` | Coriolis 与 centrifugal 广义力，不含重力和外部载荷 |
| `gravity` | 静态重力项，可选 link 局部外部载荷 |
| `inverse_dynamics` | RNEA 逆动力学；考虑关节状态、重力和可选外载 |
| `forward_dynamics` | ABA 正动力学；浮动基根状态作为显式输入 |

模型支持固定和浮动基座，以及 revolute / continuous / prismatic / fixed joints。fixed joint 不占广义坐标；浮动基的六维根运动单独作为广义量处理。浮动模型的 root link 必须具有正质量的 inertial block；DLS 逆运动学不适用于浮动基。

## 官方基准与验证

上游 README 对照 Pinocchio，报告如下加速比。模型是 Franka 固定基 7 关节机械臂和 Unitree G1 浮动基 29 关节人形机器人，Rust 与 Python 分别报告。数字是作者测量，不代表其他 CPU、编译配置或模型上的结果。

| 运算 | Rust Franka | Rust G1 | Python Franka | Python G1 |
|------|------------:|--------:|--------------:|----------:|
| Jacobian | 1.59× | 1.80× | 1.28× | 1.38× |
| RNEA | 1.74× | 1.81× | 1.17× | 1.54× |
| ABA | 1.20× | 1.14× | 1.81× | 1.89× |

复现脚本位于 `benches/`。仓库测试覆盖有限差分、生成 URDF、动力学一致性、固定与浮动基、外部载荷、solver errors、分配行为、绑定安装和独立 Pinocchio oracle 对照。测试架构说明见 [tests/TESTING.md](https://github.com/xiaojie-xue/dynibo/blob/main/tests/TESTING.md)。

## 仓库入口（2026-10-06 核查）

| 路径 | 角色 |
|------|------|
| `src/model.rs`、`src/model/`、`src/robot.rs`、`src/robot/` | Rust 模型、Robot 类型、workspace、运动学与动力学算法 |
| `src/base.rs`、`src/spatial.rs`、`src/error.rs` | BaseState、空间量和结构化错误 |
| `bindings/python/` | PyO3 Python 扩展及 Python API |
| `bindings/c/` | C ABI、C++ 包装头文件、CMake 配置 |
| `docs/user-guide/kinematics.zh.md` | 运动学 API、Jacobian、DLS-IK |
| `docs/user-guide/dynamics.zh.md` | 质量矩阵、速度乘积力、重力、RNEA 与 ABA |
| `docs/user-guide/fixed-and-floating-bases.zh.md` | 固定/浮动基座输入与维度约定 |
| `benches/` | Rust / Python 与 Pinocchio 性能基准 |
| `tests/`、`ci/test-all.sh` | 数值、内存、绑定和跨语言测试 |

## 版本与发布状态

GitHub 最新 release 为 v0.5.1。该版本移除旧 Python ctypes fallback，Python API 改为要求 PyO3 原生扩展；发布说明称 Rust 核心及 C/C++ 绑定没有变化。PyPI 页面在核查时仍显示 0.1.0（2026-08-05），安装 Python 包前应确认它是否已同步上游所需 API。

项目源码开源且采用 MIT；仓库中的 Franka / Unitree 机器人描述有独立第三方许可。没有模型权重或机器人训练数据集。

## 对 wiki 的映射

- 实体页：[Dynibo](../../wiki/entities/dynibo.md)
- 刚体动力学算法：[Articulated Body Algorithms](../../wiki/formalizations/articulated-body-algorithms.md)
- 主流全栈对照：[Pinocchio](../../wiki/entities/pinocchio.md) 与 [Pinocchio 来源归档](./pinocchio.md)
- 模型入口：[URDF](../../wiki/concepts/urdf-robot-description.md)
- IK 方法对照：[ssik](../../wiki/entities/ssik.md) — Dynibo 固定基数值 DLS 与 ssik 解析多分支
- 快速上手：[Pinocchio 快速上手](../../wiki/queries/pinocchio-quick-start.md)
- 控制应用：[重力补偿](../../wiki/concepts/gravity-compensation.md)

## 一手资料链接

- [上游 GitHub README](https://github.com/xiaojie-xue/dynibo/blob/main/README.md)
- [官方中文 README](https://github.com/xiaojie-xue/dynibo/blob/main/README.zh.md)
- [官方中文运动学指南](https://github.com/xiaojie-xue/dynibo/blob/main/docs/user-guide/kinematics.zh.md)
- [官方中文动力学指南](https://github.com/xiaojie-xue/dynibo/blob/main/docs/user-guide/dynamics.zh.md)
- [官方固定/浮动基座指南](https://github.com/xiaojie-xue/dynibo/blob/main/docs/user-guide/fixed-and-floating-bases.zh.md)
- [Release v0.5.1](https://github.com/xiaojie-xue/dynibo/releases/tag/v0.5.1)
- [Crates.io](https://crates.io/crates/dynibo)
- [PyPI](https://pypi.org/project/dynibo/)
