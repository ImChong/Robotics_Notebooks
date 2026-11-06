# 宇树官方仓 README 通读（unitree_sdk2 + unitree_rl_gym）

> 来源归档（原文通读 · 支撑 `wiki/entities/unitree.md`）

- **标题：** unitreerobotics/unitree_sdk2 与 unitreerobotics/unitree_rl_gym 仓库 README
- **机构：** 宇树科技（Unitree）
- **类型：** repo / official README
- **原文：** <https://github.com/unitreerobotics/unitree_sdk2>、<https://github.com/unitreerobotics/unitree_rl_gym>（2026-11 通过 raw.githubusercontent.com 获取 main 分支 README 全文）
- **一句话说明：** 前者给出 SDK2 的构建与安装方式，后者给出官方 RL 仓 `Train → Play → Sim2Sim → Sim2Real` 的标准流程与命令参数；合起来是宇树「SDK 底层 + RL 训练」两条主线的官方入口描述。
- **核对范围：** 两个 README 全文逐段通读（环境依赖、构建、安装、训练/回放/仿真/真机命令、参数表、C++ 部署、致谢、许可）；README 内链接的 `doc/setup_en.md`、`deploy/deploy_real/README.md`、Unitree 开发者文档中心及视频动图未读，其余 GitHub 组织仓未读。
- **通读：** 已完成

## 核心摘录（均出自原文）

- **SDK2 环境（unitree_sdk2）：** 预设环境为 Ubuntu 20.04 LTS，CPU 支持 aarch64 与 x86_64，gcc 9.4.0；依赖 CMake ≥3.10、GCC、Make，以及 libyaml-cpp、eigen3、boost、fmt 等包。
- **SDK2 构建与安装：** 在仓内 `mkdir build && cmake .. && make` 构建示例；`sudo make install` 安装到系统目录，或用 `-DCMAKE_INSTALL_PREFIX=/opt/unitree_robotics` 指定目录；装到别处需把路径加入 `CMAKE_PREFIX_PATH` 以便 `find_package()`；`example/cmake_sample` 示范如何在 CMake 工程中引入。更多信息指向 Unitree 文档中心。
- **RL 仓定位（unitree_rl_gym）：** 基于宇树机器人的强化学习实现，支持 Go2、H1、H1_2、G1；展示 Isaac Gym、MuJoCo、真机三种运行结果。
- **标准流程：** `Train → Play → Sim2Sim → Sim2Real`。Train 在 Gym 中交互求最大化奖励，不建议训练时实时可视化（降低效率）；Play 验证策略；Sim2Sim 把 Gym 训练的策略放到其他仿真器，确认不过度依赖 Gym 特性；Sim2Real 部署到实体机。
- **训练命令：** `python legged_gym/scripts/train.py --task=xxx`，task 取 go2 / g1 / h1 / h1_2；参数含 `--headless`（无界面更高效）、`--resume`、`--num_envs`、`--seed`、`--max_iterations`、`--sim_device`、`--rl_device` 等；默认结果目录 `logs/<experiment_name>/<date_time>_<run_name>/model_<iteration>.pt`。
- **Play 与导出：** Play 默认加载最近一次 run 的最新模型，并把 Actor 网络导出到 `logs/{experiment_name}/exported/policies`（MLP 为 `policy_1.pt`，RNN 为 `policy_lstm_1.pt`）。
- **Sim2Sim：** `python deploy/deploy_mujoco/deploy_mujoco.py {config_name}`，配置默认在 `deploy/deploy_mujoco/configs/`；默认模型 `deploy/pre_train/{robot}/motion.pt`，自训模型需改 YAML 中的 `policy_path`。
- **Sim2Real：** 部署前必须让机器人处于 debug 模式；`python deploy/deploy_real/deploy_real.py {net_interface} {config_name}`，net_interface 为连机器人的网卡名（如 enp3s0），配置有 g1 / h1 / h1_2。
- **C++ 部署：** 提供 G1 预训练模型的 C++ 示例（`deploy/deploy_real/cpp_g1`），依赖 LibTorch（README 给出 2.7.1 CPU 版下载），编译后 `./g1_deploy_run {net_interface}`。
- **致谢与许可：** 基于 legged_gym（训练与运行代码基础）、rsl_rl（RL 算法）、mujoco、unitree_sdk2_python（真机通信）；BSD 3-Clause，需保留版权声明、不得用项目/组织名做宣传、修改需披露。

## 对 wiki 的映射

- [`wiki/entities/unitree.md`](../../wiki/entities/unitree.md)
- [`wiki/entities/unitree-sdk2.md`](../../wiki/entities/unitree-sdk2.md)
- [`wiki/entities/unitree-rl-gym.md`](../../wiki/entities/unitree-rl-gym.md)

## 参考来源

- <https://github.com/unitreerobotics/unitree_sdk2>
- <https://github.com/unitreerobotics/unitree_rl_gym>
