# TipSim：华威科 TIP 触觉仿真集成项目（官方仓库）

> 原始资料归档（repo / MuJoCo tactile simulation，2026-11-13）

- **名称：** TipSim — 华威科 TIP 平台触觉仿真开放集成项目
- **项目/代码：** <https://github.com/tiplab/TipSim>
- **官方中文文档：** <https://github.com/tiplab/TipSim/blob/master/README.zh-CN.md>
- **官方示例说明：** <https://github.com/tiplab/TipSim/blob/master/samples/README.zh-CN.md>
- **Roadmap：** <https://github.com/tiplab/TipSim/blob/master/ROADMAP.zh-CN.md>
- **固定核验版本：** [6a323daf](https://github.com/tiplab/TipSim/tree/6a323daf4888ec3f0537ddc7d6d7675037c6e445)（上游 2026-11-05 初始提交）
- **机构：** 华威科 / TIP（据官方 README）
- **入库日期：** 2026-11-13
- **开源状态：** **部分开放**。XML 场景、Python/C++ 示例、CMake 和文档为 MIT；`lib/*.so` 触觉插件仅二进制提供，**不在 MIT 授权范围内**，未经书面约定不得反编译、逆向或脱离本仓库单独再分发。
- **通读：** 已完成
- **核对范围：** 通读固定版本官方 `README.zh-CN.md`（介绍、效果、参数表、安装依赖、插件加载、XML 配置、触觉读取、许可），核对 `samples/README.zh-CN.md` 的复现入口与依赖、`ROADMAP.zh-CN.md` 的阶段规划；逐段检查 `samples/tip/python/universa_pick_and_place.py` 中 `main/Robot.step/Robot.force_close/pick_place` 的入口及双侧合力判断，`samples/tip/python/loong96.py` 的阵列读取，并核对 `samples/common/tactile_ros2.py` 发布接口与 C++ 对应示例的入口声明。**未运行** MuJoCo/ROS2，**未取得**插件内部实现，也未复现实验或进行真机 Sim2Real 测试。
- **一句话：** 以可加载的 MuJoCo 预编译触觉插件、阵列场景和 Python/C++ 示例，演示触觉接触观测及基于阈值的抓取控制。

## 主要原始资料核对

| 来源 | 已核实范围 | 边界 |
|---|---|---|
| [官方 README.zh-CN.md](https://github.com/tiplab/TipSim/blob/6a323daf4888ec3f0537ddc7d6d7675037c6e445/README.zh-CN.md) | Loong、Universa 两个插件；`mj_loadPluginLibrary` → 加载 XML → `mj_step/mj_forward` → `sensordata`；`size/npoints/coords` 与数值特性配置 | 并未公开插件内部力分配算法源码 |
| [samples/README.zh-CN.md](https://github.com/tiplab/TipSim/blob/6a323daf4888ec3f0537ddc7d6d7675037c6e445/samples/README.zh-CN.md) | Python/C++ 场景及运行命令、ROS2 消息输出、构建依赖 | 运行示例需额外 Python/ROS2 或 C++/GLFW 等依赖 |
| [universa_pick_and_place.py](https://github.com/tiplab/TipSim/blob/6a323daf4888ec3f0537ddc7d6d7675037c6e445/samples/tip/python/universa_pick_and_place.py) | UR5e + Universa 双指；`force_close` 的分指受力阈值；A 箱抓取 → B 箱放置 | 抓取是规则阈值控制，不等于学习策略或真机成功率 |
| [loong96.py](https://github.com/tiplab/TipSim/blob/6a323daf4888ec3f0537ddc7d6d7675037c6e445/samples/tip/python/loong96.py) | Loong 12×8 阵列，打印 `total/active`；`TactileRos2Publisher` | 示例只证明可读取、发布，不代表硬件采样带宽 |
| [LICENSE](https://github.com/tiplab/TipSim/blob/6a323daf4888ec3f0537ddc7d6d7675037c6e445/LICENSE) | MIT 明确排除 `lib/*.so` | 不可将整个项目称作“核心完全开源” |
| [ROADMAP.zh-CN.md](https://github.com/tiplab/TipSim/blob/6a323daf4888ec3f0537ddc7d6d7675037c6e445/ROADMAP.zh-CN.md) | 当前 MuJoCo、两款传感器；拟扩展 Isaac Sim、数据集采集与 Benchmark | 规划内容**不是已实现功能** |

## 可核对的关键行为

1. **输入与阵列：** Loong 使用 `size` 配置行列，Universa 使用 `npoints` 配置点数；`coords` 指定局部 (y,z) 触点坐标，单位 mm。示例有 Loong 96 点和 Universa 80 点。
2. **输出与控制：** 插件输出经标准 `mjData.sensordata` 取得。取放脚本每个物体的目标受力 `target=10.0 N`；至少达到 `MIN_GRIP=10.0` 指令，且双指**各自**满足 `target/2`，或达到最大行程 `GRIP_MAX=255` 后停止增大闭合指令。不是“合力达到阈值便无条件停止”的简化逻辑。
3. **设备特性：** 官方文档列 `full_scale`（Loong 默认 20 N、Universa 30 N，阵列合力量程）、`accuracy=1.0 %FS`、`repeatability=0.7 %FS`、`trigger_force=0.05 N`、迟滞参数和随机种子。此处是**仿真配置参数**，不能视作真机测得性能。
4. **版本：** Linux x86_64；插件兼容 MuJoCo 3.9.0–3.16.0；官方样例以 3.12.0 推荐配置。
5. **开放与复现：** 可以获得插件二进制及示例，但无法审计其内部算法；本文没有独立运行性能基准，也不宣称零 sim-to-real gap。

## 需要谨慎描述

- “行业首个”“多模态”均不能从这套材料独立验证。当前公开的主要输出为**阵列触点力值**，不能把它等同于同时开放视觉、温度、剪切力等多模态接口。
- 官方 GIF/视频演示不构成真机成功率、泛化或可迁移性证据。
- Isaac Sim、自动采数与 Benchmark 属路线图，而不是当前交付承诺。

## 对 wiki 的映射

- [TipSim 实体页](../../wiki/entities/tipsim.md)
- [触觉感知](../../wiki/concepts/tactile-sensing.md)
- [接触丰富操作](../../wiki/concepts/contact-rich-manipulation.md)
