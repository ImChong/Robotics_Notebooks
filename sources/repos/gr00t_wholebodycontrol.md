# GR00T-WholeBodyControl — 原始资料归档

- **来源**：https://github.com/NVlabs/GR00T-WholeBodyControl
- **类型**：repo
- **机构**：NVIDIA NVlabs（GEAR WBC）
- **归档日期**：2026-05-14
- **GitHub**：约 2.0k stars；**代码** Apache-2.0，**模型权重** 另受 NVIDIA Open Model License 等约束（见仓库 `LICENSE` / `legal/`）

## 一句话说明

**GR00T Whole-Body Control (WBC)** 单仓聚合 **解耦 WBC**（GR00T N1.5 / N1.6 等使用的 RL 下肢 + IK 上肢）、**GEAR-SONIC**（规模化运动跟踪通用人形控制）与 **MotionBricks**（实时潜空间生成式运动）的训练、评测与部署脚本及文档入口。

## 为什么值得保留

- 官方 **Isaac Lab 2.3.2** 对齐徽章与 [GitHub Pages 文档](https://nvlabs.github.io/GR00T-WholeBodyControl/)，覆盖安装、训练、ZMQ/VR 遥操作、VLA 数据链路与真机 C++ 部署
- **GEAR-SONIC**：SONIC 训练代码、HF 权重、C++ 推理、VR 采集与 [在线交互 Demo](https://nvlabs.github.io/GEAR-SONIC/demo.html)（集成 Kimodo 文生运动）
- **MotionBricks**：子目录预览版（交互 G1 Demo、VQVAE/pose/root 权重、合成训练与表示文档），与 [MotionBricks 论文页](https://nvlabs.github.io/motionbricks/) 对应
- 工程注意事项明确：**Git LFS** 拉取大资源；多用途 **独立 venv**（训练 / MuJoCo / 遥操作 / 数据采集）

## 仓库结构（README 摘要）

| 组件 | 职责 |
|------|------|
| `gear_sonic` | SONIC 训练（PPO、数据处理、Hydra 配置） |
| `gear_sonic_deploy` | 真机 C++ 推理栈 |
| `motionbricks` | MotionBricks 预览代码与资源 |

## 对 wiki 的映射

1. **[MotionBricks](../../wiki/methods/motionbricks.md)**：生成式运动层与本仓 `motionbricks/` 子项目对应
2. **[SONIC（规模化运动跟踪）](../../wiki/methods/sonic-motion-tracking.md)**：GEAR-SONIC 论文与训练/部署入口
3. **[Foundation Policy / GR00T](../../wiki/concepts/foundation-policy.md)**：N1.5 / N1.6 解耦 WBC 与 VLA 工作流文档（站点 tutorials）
4. **[GR00T-VisualSim2Real](../../wiki/entities/gr00t-visual-sim2real.md)**：同品牌下视觉 Sim2Real 另一官方仓库，分工不同

## 引用（GEAR-SONIC，仓库 README）

```bibtex
@article{luo2025sonic,
  title={SONIC: Supersizing Motion Tracking for Natural Humanoid Whole-Body Control},
  author={Luo, Zhengyi and others},
  journal={arXiv preprint arXiv:2511.07820},
  year={2025}
}
```
