# MotionBricks — NVIDIA 官方项目页

- **来源**：https://nvlabs.github.io/motionbricks/
- **类型**：site（项目页 / SIGGRAPH 2026 展示）
- **机构**：NVIDIA Research
- **归档日期**：2026-09-15
- **论文**：arXiv:2604.24833 — *MotionBricks: Scalable Real-Time Motions with Modular Latent Generative Model and Smart Primitives*
- **会议**：ACM Transactions on Graphics · SIGGRAPH 2026
- **代码（步骤 2.5）**：**部分开源** — 预览版在 [GR00T-WholeBodyControl/motionbricks](https://github.com/NVlabs/GR00T-WholeBodyControl/tree/main/motionbricks)（交互 G1 Demo + 合成训练管线 + BONES-SEED 接入说明）；**完整版**（深度嵌入 GR00T WBC 的机器人 formulation + 完整训练管线）项目页称约一个月后发布
- **社区移植**：[localai-org/motion-bricks.cpp](https://github.com/localai-org/motion-bricks.cpp)（C++/GGML G1 推理，非 NVIDIA 官方）

## 一句话说明

**MotionBricks** 是 NVIDIA 的大规模 **实时生成式运动框架**：单一神经网络骨干覆盖 **350,000+** 运动片段，报告 **~15,000 FPS / ~2 ms** 延迟，并通过 **Smart Primitives** 为导航与物体交互提供统一、可零样本组合的高层接口；已作为 **GR00T Whole-Body Control** 的运动生成层。

## 为什么值得保留

- 项目页展示 **2:40 UE5 无剪辑 Demo**（全神经生成、无 foot-locking/blending/手工过渡）与 **Smart Locomotion / Smart Objects** 交互范式
- 明确 **Code & Data Release** 策略：预览代码并入 GR00T WBC 单仓，而非独立 GitHub 组织
- **Unitree G1** 人形演示与 **GEAR-SONIC** 跟踪生态互链

## 核心能力（项目页归纳）

| 模块 | 要点 |
|------|------|
| Modular Latent Backbone | 单一模型建模 350k+ clips；模块化潜空间生成 |
| Smart Locomotion | 速度/朝向/风格命令零样本组合（injured、zombie、skipping、strafing 等） |
| Smart Objects | 代理关键帧定义交互；骨干填充 approach/contact/follow-through |
| 性能 | ~15,000 FPS、~2 ms latency（论文/项目页口径） |
| 机器人 | G1 交互 Demo；GR00T WBC 运动意图层 |

## 对 wiki 的映射

1. **[MotionBricks（方法页）](../../wiki/methods/motionbricks.md)** — 技术主入口
2. **[paper-motionbricks（论文实体）](../../wiki/entities/paper-motionbricks.md)** — 论文结论与开源边界
3. **[motion-bricks.cpp（社区实体）](../../wiki/entities/motion-bricks-cpp.md)** — C++/GGML 部署档
4. **[GR00T-WholeBodyControl](../repos/gr00t_wholebodycontrol.md)** — 官方预览代码仓

## 关联原始资料

- [MotionBricks 论文摘录](../papers/motionbricks.md)
- [GR00T-WholeBodyControl 仓](../repos/gr00t_wholebodycontrol.md)
