# steerable-policies-bridge

- **仓库地址：** <https://github.com/steerable-policies/steerable-policies-bridge>
- **许可证：** MIT
- **默认分支：** `master`
- **核对方式：** 静态代码核对（基于仓库目录采样与 README 文档，未本地运行）
- **关联论文：** [paper-agentic-2602-13193](../papers/paper-agentic-2602-13193.md)

## 仓库定位与依赖边界

`steerable-policies-bridge` 是论文 *Steerable Vision-Language-Action Policies for Embodied Reasoning and Hierarchical Control* 的官方 OpenVLA 基座训练与评测仓库，基于 OpenVLA / Prismatic VLM 代码框架改造而来。仓库本身聚焦 Bridge 数据集上的 BC 训练与高阶具身推理器（Embodied Reasoner）微调；WidowX 实机客户端推理与 Gemini 上下文高层控制脚本则通过配套仓库 [`steerable-policies/steerable-gym`](https://github.com/steerable-policies/steerable-gym)（适配自 Manipulator Gym）配合 AgentLace 与 Interbotix ROS1 完成。

已验证公开发布的关联资产：
- **策略权重：** [`Embodied-CoT/steerable-policy-openvla-7b-bridge`](https://huggingface.co/Embodied-CoT/steerable-policy-openvla-7b-bridge)
- **合成转向标注集：** [`Embodied-CoT/steering_features_bridge`](https://huggingface.co/datasets/Embodied-CoT/steering_features_bridge)
- **未在本仓库涵盖：** README 中 Embodied Reasoner 独立权重链接处于注释占位状态；论文附录 F-A2 所述的 $\pi_{0.5}$ 版 Steerable Policy 基于 OpenPi 微调，未纳入本仓库。

## 核心目录与入口文件（静态核对）

| 路径 | 角色与机制 |
|---|---|
| `prismatic/vla/datasets/datasets.py` | 核心数据改动所在：定义 `RLDSBatchTransform` 与 `ReasonerRLDSBatchTransform`，通过 `PATH_TO_REASONING_DATA` 索引 `steering_features_bridge` 字典；前者随机采样帧级转向指令替换默认任务文本，后者构造“推理依据 + 转向指令”自回归监督目标 |
| `vla-scripts/train.py` | 主训练入口；启用 `train_reasoner` 配置项可从同一套 OpenVLA 管线切换为高阶具身推理器微调 |
| `vla-scripts/finetune.py` | VLA 微调脚本（继承自 OpenVLA） |
| `vla-scripts/deploy.py` | 策略推理服务部署入口 |
| `robot/bridge/run_bridgev2_eval.py`、`robot/bridge/widowx_env.py`、`robot/bridge/bridgev2_utils.py` | BridgeV2 / WidowX 环境评测与工具封装 |
| `robot/openvla_utils.py`、`robot/robot_utils.py` | OpenVLA 推理与通用机器人控制工具 |
| `robot/libero/run_libero_eval.py`、`robot/simpler/run_simpler_eval.py` | 继承自 OpenVLA 的 LIBERO / SimplerEnv 评测脚本（注：论文 Appendix E-D 指出 LIBERO 因轨迹单峰且初始状态随机化不足而不适合作为 Steerable Policies 主训练场） |

## 官方运行链路摘要（据 README 与代码入口）

1. **训练阶段：** 下载 `steering_features_bridge` 后修改 `prismatic/vla/datasets/datasets.py` 中的 `PATH_TO_REASONING_DATA`，按 OpenVLA FSDP 全分片训练命令运行 `vla-scripts/train.py`；若训练高阶推理器则打开 `train_reasoner`。
2. **WidowX 实机推理（双机/服务端-客户端架构）：**
   - 机器人主机启动 `interbotix_xsarm_control` ROS 节点，并在 `steerable-gym` 下运行 `manipulator_server.py --widowx --cam_ids 0`。
   - 策略主机启动 `policy_server.py`（默认监听 `0.0.0.0:8000/act`）。
   - 根据 `steerable-gym/policies/command.yaml` 中的 `policy_mode`（`human` / `gemini` / `task_level`）运行 `policies/steerable_eval.py`，或另启 `reasoner_server.py` 后运行 `policies/embodied_eval.py --requery_rate 5` 实现高阶具身推理器闭环控制。
