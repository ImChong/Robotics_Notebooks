# Isaac Sim 官方 README 与文档首页通读

> 来源归档（原文通读 · 支撑 `wiki/entities/isaac-sim.md`）

- **标题：** isaac-sim/IsaacSim README（main 分支）与 Isaac Sim 官方文档首页「What Is Isaac Sim」
- **作者 / 来源：** NVIDIA
- **类型：** repo README / official docs landing page
- **原文：** <https://github.com/isaac-sim/IsaacSim>（README 全文）；<https://docs.isaacsim.omniverse.nvidia.com/latest/index.html>（文档首页全文，2026-11 抓取）
- **一句话说明：** Isaac Sim 是建在 Omniverse 上的机器人仿真平台，用于在逼真虚拟环境中开发、测试、训练与部署 AI 机器人。
- **核对范围：** 通读 README 全文（简介、Key Features / Applications、前置条件与 GPU 表、源码构建 `build.sh`、构建选项、二进制 / pip wheel / Docker 打包、排障、Support、License、Citation、Contributing）与文档首页全文（安装入口、教程分级、Workflow Overview 三类流程）。首页链接到的各教程章节、Core API 参考、Docker README 未逐页阅读。
- **通读：** 已完成

## 核心摘录（均出自原文）

- **定位（README）：** 基于 NVIDIA Omniverse 的仿真平台，用来开发、测试、训练、部署 AI 机器人；支持从 URDF、MJCF、CAD 等格式导入；用 GPU 加速物理引擎和多传感器 RTX 渲染；自带合成数据生成、强化学习、ROS 集成、数字孪生等端到端工作流。
- **Key Features：** 资产导入导出（非 USD 格式互转）、Robot Tuning（按物理精度 / 计算效率 / 真实感优化机器人）、Robot Simulation（控制器、运动生成、运动学求解、策略集成）、Sensors（RTX 与物理传感器）。
- **Key Applications：** Isaac Lab（GPU 加速的强化学习 / 模仿学习 / 运动规划框架）、ROS Bridge、Synthetic Data Generation。
- **环境要求：** Windows 11 或 Ubuntu 22.04/24.04；Python 3.12 徽章；本地工作站 GPU 最低 RTX 4080、推荐 RTX 5080 / RTX 5880 Ada，数据中心最低 A40、推荐 L40S / L20；Linux 构建需 GCC/G++ 11，更高版本不受支持；需访问 Kit 扩展仓库域名。
- **源码构建：** `git clone` + `git lfs pull`，`./build.sh`（Windows `build.bat`），产物在 `_build/<platform>/release`，运行 `isaac-sim.sh`；首次启动因扩展与 shader 缓存可达数分钟，之后约 10–30 秒。
- **打包：** `repo.sh package` 生成独立二进制包；`python_package --create / comment_archive_deps / --wheel` 三步生成 pip wheel（预编译版本见 pypi.nvidia.com）；Docker 见 tools/docker。
- **许可：** Apache 2.0；外部贡献范围有限，需先读 CONTRIBUTING。
- **文档首页：** 安装入口为 Quick Install / Workstation / Container / Python（pip、conda）；教程分初级（UI、Python 脚本、URDF 导入）、中级（ROS 2 TurtleBot、Replicator SDG、ROS 2 仿真控制）、高级（为 Isaac Lab 准备机器人、AMR 导航合成数据、经 ROS 2 运行 RL 策略）。
- **Workflow Overview：** 技术组件为 RTX、Newton、PhysX、OmniGraph、Replicator、OpenUSD；流程为导入 → 配置机器人与场景 → 仿真 → 连接外部栈；各阶段都作用于同一个共享场景。SDG 路径：标注场景、变化条件、仿真、渲染传感器输出；SIL 路径：配置机器人物理、传感器与通信图，在上硬件前验证外部机器人栈。

## 对 wiki 的映射

- [`wiki/entities/isaac-sim.md`](../../wiki/entities/isaac-sim.md)
