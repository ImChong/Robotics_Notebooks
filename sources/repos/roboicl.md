# RoboICL 官方代码与运行入口核对

- **代码：** https://github.com/Mosi-AI/RoboICL
- **官方项目：** https://mosi-ai.github.io/RoboICL-GPT6-Astra.github.io/
- **论文：** https://arxiv.org/abs/2609.34261
- **许可：** 项目自有代码 MIT；第三方子模块沿用各自许可。
- **核对日期：** 2026-11-21
- **核对范围：** 主仓 README 中安装、锁文件、协议与运行命令；roboicl/run.py、roboicl/rollout.py、roboicl/policy_server.py 以及 roboicl/policy 目录索引。
- **通读：** 已完成

## 已确认的可运行路径

1. ```git clone --recurse-submodules https://github.com/Mosi-AI/RoboICL.git```。
2. 设置独立的 ```ROBOICL_SIM_PYTHON```、```ROBOICL_POLICY_PYTHON``` 及数据根目录；Isaac Sim 5.1 兼容环境与 NVIDIA GPU 是前提。
3. 运行 setup/fetch_assets.py 与 setup/fetch_reference.py；锁定 RoboDojo 数据 SHA256 与一示范 12 个原始时间窗口。
4. ```python -m roboicl.run deposit_coin --shots 0 --seed 0 --layout 0 --profile configs/protocols/zero_shot_b25.json --dry-run``` 仅预检查，不调用 API。
5. 同一命令增加 ```--capture-only``` 可确认摄像机流，不执行模型动作。
6. 设置 ```ASTRA_API_KEY``` 后运行 zero_shot_b25 或 one_shot_j12_b12 协议，成功/进度看输出的 result.json；进程退出码 0 只表示评测流程完成，并不代表任务成功。

## 源码模块对齐

- ```roboicl/run.py```：验证依赖、任务和参考数据，编排策略/仿真进程。
- ```roboicl/policy_server.py```：XPolicyLab 的 WebSocket PolicyServer 承载策略 Model。
- ```roboicl/rollout.py```：env.get_obs → policy get_action → env.take_action → 更新 observation/receipt → finalize。
- ```roboicl/policy/dialogue_policy.py```：交互政策、模型与记忆处理；```configs/protocols/``` 保存 TRAIN/LIVE 预算。
- 注意：这里只核验源码和使用方式，**未在本环境安装 Isaac Sim 或执行机器人实测**。

## 对 wiki 的映射

- [RoboICL](../../wiki/entities/paper-roboicl.md)
