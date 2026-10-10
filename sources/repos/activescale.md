# ActiveScale 官方训练、推理与遥操作源码

- **类型：** repo
- **代码：** <https://github.com/ShuaiZhou302/ActiveScale>
- **核查日期：** 2026-10-10
- **实际克隆快照：** `9bfd6a2fb3f7d505bd8bcd5bf90136ed270df9f4`
- **论文：** <https://arxiv.org/abs/2609.18514>；[归档](../papers/activescale_arxiv_2609_18514.md)
- **项目页：** <https://active-scale.github.io/>；[数据/模型核查](../sites/activescale.md)
- **开放状态：** 训练、远程推理、RTC、数据检查与 Quest 2 遥操作源码已公开，非 README 占位。
- **许可：** 代码 Apache-2.0；Gemma 派生权重另受 Gemma 条款约束。

## 核查文件与真实路径

| 文件/目录 | 作用 |
|---|---|
| `scripts/train_activescale.sh` → `scripts/train_pytorch.py` | smoke/midtrain/posttrain 配置，通过 torchrun 启动；默认 8 进程并非单卡要求 |
| `docs/TRAINING.md`、`configs/activescale.example.env` | 路径与 checkpoint、H50、四历史 slot、1:1、bf16、目标权重 |
| `docs/DATA_FORMAT.md`、`scripts/validate_activescale_data.py` | 数据 schema、参考系、32D 容器与 mask；schema 不代替 loader 检验 |
| `src/openpi/training/piper_camera_token_dataset.py`、`public_robot_cotrain_dataset.py` | Piper 与公开机器人数据适配 |
| `src/openpi/models_pytorch/pi0_pytorch.py`、`camera_head.py`、`rtc.py` | 历史/token、几何辅助监督、普通/RTC 采样 |
| `scripts/serve_policy.py` → `src/openpi/policies/policy_config.py` | --config/--checkpoint 建策略，加载权重与归一化资产，默认 flow 10 步 |
| `src/openpi/serving/websocket_policy_server.py` → `src/openpi/policies/policy.py` | 请求转 infer，归一化、采样、反归一化，返回 actions/infer_ms |
| `packages/openpi-client` | 远程协议，不代表完整自主真机部署器 |
| `teleoperation/quest2` | 输入、pose server、三相机流、IK、安装板 STEP；需本地 URDF/ROS 驱动及校准 |

## 复现与风险

README 指 Python 3.11、torch 2.7.1、特定 LeRobot commit 与 transformers patch；配置数据/缓存/资产/checkpoint。JAX 基座中训先用 `convert_jax_model_to_pytorch.py`，不把 JAX 目录当 PyTorch 权重。

布局 `[left pose7, right pose7, camera pose7, gripper2, masked padding9]`；人类 pack 会重排顺序。不同来源独立 q01/q99，loader 拒绝布局不匹配；缺失头位姿填零会产生错误监督。

RTC 需 PyTorch 支持、旧块 `obs['actions']`、延迟与 execution horizon，不是开服务器就自动异步真机闭环。相机历史、IO、安全限制需实际客户端集成。

Quest README 有断电标定、低速试机、物理急停要求，watchdog/base-stop 不代替硬件安全。本次只克隆和审阅源码/说明，未执行安装、训练、服务或机器人控制。

## 对 wiki 的映射

- [ActiveScale 唯一实体](../../wiki/entities/paper-activescale.md) — 时序图、32D/23D 与开放状态，不建 repo/wiki 副本。
