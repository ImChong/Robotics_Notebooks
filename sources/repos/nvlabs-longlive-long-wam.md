# NVlabs/LongLive 中的 Long-WAM 官方代码

> 来源归档（repo；核对 Long-WAM README、VERIFICATION、BENCHMARKS、infra 文档及模型集合；2026-10-08）

- **仓库：** <https://github.com/NVlabs/LongLive>
- **项目子目录：** <https://github.com/NVlabs/LongLive/tree/main/Long-WAM>（代码范围 `Long-WAM/`）
- **项目页：** <https://nvlabs.github.io/LongLive/Long-WAM/>
- **论文：** <https://arxiv.org/abs/2610.10528>
- **权重：** <https://huggingface.co/collections/Efficient-Large-Model/long-wam>
- **License：** Long-WAM 源码 Apache-2.0；第三方组件、基础模型、数据和模拟器资产按各自许可。
- **仓库边界：** LongLive monorepo 同时包含视频生成与 Long-WAM 研究代码。Long-WAM 是单独维护的项目子树，具有独立模型、配置、训练/评测入口和运行时。

## 实现与工作流

- README 提供模型/配置、训练和 benchmark 评估入口；各任务使用独立脚本/配置。
- 评测涉及 LIBERO、RoboTwin 2.0、DOMINO、RoboCasa GR-1 / RoboCasa365；LIBERO 与 RoboTwin 有 IDM、CodeDenoise 等动作路径。
- 以长观测历史预测未来视觉 latent，再由动作专家根据历史、未来表征与机器人条件生成动作块。评估历史长度包括 0、2.4、4.8、9.6、19.2、38.4 秒。
- 异步部署将模型推理和环境/机器人动作执行流水化；包含 LeRobot policy interface，以及 LIBERO、RoboTwin 2、YAM、Franka、Unitree G1 等适配或转换。
- 文档列出 RTX 5090、DGX Spark、Jetson AGX Thor 部署/加速路径。单个 GR-1 检查点卡示例采用 20 Hz、chunk 16、单路 ego RGB、58 维状态和 29 维动作，不代表全部模型。

## 验证记录

官方 `Long-WAM/docs/VERIFICATION.md` 记录源快照上 87 项 CPU 回归测试和 41 项 CPU-only CLI 检查通过；视频/CUDA 测试有跳过。记录未运行 GPU job、完整模拟器 rollout 或物理机器人命令；环境也不是按目标依赖锁重新安装。官方 `Long-WAM/docs/BENCHMARKS.md` 将完整 benchmark reproduction 标为未验证。

CPU 测试不能验证论文成功率或 RTX 5090 时延。此归档没有启动机器人，也没有声称复现基准。

## 沉淀到 Wiki

- [Long-WAM 独立详情节点](../../wiki/entities/paper-long-wam-scaling-context.md)
- [论文来源归档](../papers/long-wam-arxiv-2610-10528.md)
- [项目页来源归档](../sites/long-wam-project-page.md)
