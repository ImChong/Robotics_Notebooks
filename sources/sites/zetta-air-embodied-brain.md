# Zetta ζ 项目页（air-embodied-brain.github.io/zetta）

> 来源归档

- **标题：** Zetta ζ: An Efficient Closed-Loop Embodied Harness for Self-Evolving Physical Intelligence
- **类型：** site（项目页）
- **链接：** https://air-embodied-brain.github.io/zetta/
- **论文：** [2608.16590](https://arxiv.org/abs/2608.16590)
- **代码：** [Zetta-Embodiment](../repos/air-embodied-brain-zetta-embodiment.md)
- **入库日期：** 2026-09-26
- **一句话说明：** 清华 AIR / Z-Trans AI 的 **闭环具身 harness**：动作频率 **critic–recovery**、三时间尺度自进化、**Z-Infra** 异构 rollout；冻结 VLA 下 LIBERO-Pro / RoboCasa SOTA 档数字与案例视频（CoffeeSetupMug、LIBERO-Pro 推盘等）。

## 开源核查（2026-09-26）

| 项 | 状态 |
|----|------|
| GitHub | **已开源** <https://github.com/air-embodied-brain/Zetta-Embodiment>（LIBERO/RoboCasa/ManiSkill/RoboTwin/Genie Sim 等集成；权重与 sim 资产外置） |
| HF | [papers/2608.16590](https://huggingface.co/papers/2608.16590)（论文元数据页，非权重仓） |
| Docker | README 提供 LIBERO-Pro / RoboCasa 预构建镜像（百度网盘链接） |

## 核心摘录（项目页）

###  headline 数字（项目页）

- LIBERO-Pro Goal 平均 **90.8%**；RoboCasa 18 任务 **93.6%**
- 相对 frozen VLA：**+56.3 pt**（LIBERO-Pro）、**+20 pt**（RoboCasa）量级（页内表述）
- **20.6×** 有效 rollout 吞吐；**11.1×** 推理加速 vs RPent（文内对照）

### 三 Loop（无 VLA 微调）

1. **Action · Critic-Governed Action Loop**
2. **Rollout Batch · Candidate Optimization Loop**
3. **Iteration · Validation-Gated Skill Update Loop**

### 案例

- **RoboCasa CoffeeSetupMug：** Round0 **70%** → Round1 **86%**（接触/抓取稳定 critic）
- **LIBERO-Pro Push plate：** Round0 **0%** → Round2 **95%**（分层 critic/recovery 链）

## 对 wiki 的映射

- [paper-zetta](../../wiki/entities/paper-zetta.md)
- 同组织对照：[Zeva 项目页](zeva.md)（因果记忆 ICL vs 代码 harness 进化）
