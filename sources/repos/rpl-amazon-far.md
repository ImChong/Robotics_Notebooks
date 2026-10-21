# RPL（Amazon FAR 人形感知行走官方实现）

> 来源归档

- **标题：** RPL: Learning Robust Humanoid Perceptive Locomotion on Challenging Terrains
- **类型：** repo / humanoid / perceptive locomotion / RL / distillation / sim2sim
- **来源：** Amazon FAR（Frontier AI & Robotics）
- **链接：** <https://github.com/amazon-far/RPL>
- **许可证：** Apache-2.0
- **配套论文：** [arXiv:2602.03002](https://arxiv.org/abs/2602.03002)
- **项目页：** <https://rpl-humanoid.github.io/>
- **入库日期：** 2026-10-21
- **一句话说明：** RPL 官方代码库，基于 Holosoma 提供分地形 RL 专家训练、多视角深度学生的 DAgger 蒸馏，以及 MuJoCo sim2sim 验证流程。
- **沉淀到 wiki：** [`wiki/entities/paper-rpl-robust-humanoid-perceptive-locomotion.md`](../../wiki/entities/paper-rpl-robust-humanoid-perceptive-locomotion.md)

## 仓库结构与能力

| 路径 | 用途 |
|------|------|
| `thirdparty/holosoma/` | Holosoma git submodule：训练核心、多深度渲染、PPO-MA、DAgger-MA 与网络模块 |
| `rpl_training/` | loco-manipulation 环境、manager terms、配置、训练入口与测试 |
| `rpl_training/data/models/` | 随仓库提供的地形专家策略 |
| `rpl_sim2sim/` | MuJoCo 仿真端与深度学生推理端的 sim2sim 示例 |

## 复现入口（README）

环境面向 Linux + NVIDIA GPU；克隆时需包含子模块：

```bash
git clone --recurse-submodules https://github.com/amazon-far/RPL.git
cd RPL
```

| 阶段 | 官方入口 | 说明 |
|------|----------|------|
| 环境 | `bash rpl_training/training_runs/setup_env.sh` | 创建 IsaacGym 训练环境 `hsgym` |
| 专家训练 | `NGPUS=4 EXPERT=stones bash rpl_training/training_runs/run_expert.sh --training.name stones-expert` | README 还列出 `stairs` 与 `slopes` 专家；建议 4 GPU |
| 学生蒸馏 | `NGPUS=8 bash rpl_training/training_runs/run_distill.sh --training.name rpl-student` | 默认加载仓库附带的 teachers；可指定自有 checkpoint；导出 ONNX |
| sim2sim | `bash rpl_sim2sim/setup_env.sh`，随后分别运行 `run_sim.sh` 与 `run_policy.sh` | MuJoCo 仿真与 `holosoma_inference` 策略作为两个进程运行；通过 Unitree SDK2 Python（DDS）通信，深度图走共享内存 |

## 开源边界与注意事项

- README 将其标为 **research release only**；README 明确说明 RPL 功能合入 Holosoma 主线属于单独工作。
- 训练基于 IsaacGym，README 指出 IsaacSim API 尚不支持其力扰动需求。
- sim2sim policy 可加载蒸馏训练导出的 `model_XXXXX.onnx`；运行 MuJoCo 示例需按文档分别准备仿真和推理环境。

## 关联资料

- 项目页归档：[`sources/sites/rpl-humanoid-github-io.md`](../sites/rpl-humanoid-github-io.md)
- 论文摘录：[`sources/papers/rpl_arxiv_2602_03002.md`](../papers/rpl_arxiv_2602_03002.md)
- Wiki 实体：[`wiki/entities/paper-rpl-robust-humanoid-perceptive-locomotion.md`](../../wiki/entities/paper-rpl-robust-humanoid-perceptive-locomotion.md)
