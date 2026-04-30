---
type: method
tags: [rl, adaptation, world-model, transformer, locomotion]
status: complete
updated: 2026-04-27
related:
  - ./model-based-rl.md
  - ./beyondmimic.md
  - ../concepts/domain-randomization.md
sources:
  - ../../sources/papers/motion_control_projects.md
summary: "Any2Track 与 RGMT 代表了基于 Transformer 和世界模型的自适应运动控制范式，通过显式建模历史扰动与未来指令来实现鲁棒的轨迹跟踪。"
---

# Any2Track & RGMT: 增强型自适应与时序建模

在复杂的人形机器人控制任务（如多步动作模仿或动态越障）中，传统的单帧马尔可夫决策过程 (MDP) 往往不足以捕捉环境的非平稳性（如地面滑移、外力扰动）。**Any2Track** 与 **RGMT** 代表了通过引入 Transformer 和世界模型来增强策略自适应能力的先进技术路线。

## RGMT: 基于 Transformer 的 Actor 网络

**RGMT** (Real-to-Game Motion Tracking) 强调了对历史信息和未来指令的重新编码。

### 1. 核心架构
- **历史编码器 (History Encoder)**：使用**因果 Transformer (Causal Transformer)** 处理最近 $K$ 步的本体感知数据。
    - **位置编码**：加入正弦位置编码以保留时间顺序。
    - **自注意力**：提取运动趋势和接触状态特征。
    - **池化**：通过最大池化得到固定长度的“动态特征向量”。
- **指令编码器 (Instruction Encoder)**：
    - **交叉注意力 (Cross-Attention)**：以动态特征向量作为查询 (Query)，从未来的指令窗口（参考关节位置等）中有选择地聚合信息。
    - **作用**：根据机器人当前的物理状态，自动筛选出最相关、最可行的指令片段。

### 2. 动作输出 (Action Output)
不同于传统的绝对角度预测，RGMT 预测**残差关节位置指令**：
$$
q_{target} = q_{ref} + \text{actor}(\text{obs}, \text{instruction\_feat})
$$

## Any2Track: 世界模型驱动的两阶段自适应

**Any2Track** 进一步将“识别扰动”与“调整动作”在架构上进行了解耦。

### 1. 两阶段训练流程
1. **第一阶段：基础技能学习**
   - 在理想/标准环境下训练一个基础动作策略。
2. **第二阶段：适配器与世界模型训练**
   - **冻结**基础动作策略。
   - **历史编码器**：观察长时序历史，输出扰动特征（Latent Representation）。
   - **世界模型 (World Model)**：辅助任务。利用扰动特征预测未来 $N$ 步的状态。预测越准，说明扰动特征提取越有效。
   - **适配器 (Adapter)**：根据扰动特征调整基础策略的输出。

### 2. 部署优势
在实际部署时，**世界模型被丢弃**，仅保留轻量级的历史编码器和适配器。这种架构使得机器人能够在不改变核心技能逻辑的情况下，通过“外挂”适配层快速适应滑移地面或负载变化。

## 主要技术路线

| 模块 | 关键技术 | 优势 |
|------|---------|------|
| **时序特征提取** | 历史编码器 (History Encoder) | 从长历史序列中识别隐式物理参数与扰动 |
| **自适应机制** | 世界模型 (World Model) 辅助训练 | 通过预测任务强制模型理解环境交互规律 |
| **策略调节** | 冻结基座 + 残差适配器 (Adapter) | 保持核心技能不变的同时快速适应新工况 |
| **输入增强** | Transformer 交叉注意力 | 实现“当前状态”与“未来指令”的最优匹配 |

## 关键技术总结：Any2Track 的残差映射

Any2Track 采用了一种自适应缩放的残差映射方式，提高动作的准确性：
1. **Tanh 映射**：将策略输出映射到 $[-1, 1]$。
2. **关节自适应缩放**：引入超参数向量 $\alpha$，为髋、膝、踝等不同关节设置不同的活动范围缩放因子。
3. **指令偏移**：$q_{target} = q_{ref} + \alpha \cdot \tanh(\text{action})$。

## 参考来源

- [sources/papers/motion_control_projects.md](../../sources/papers/motion_control_projects.md) — 飞书公开文档《开源运动控制项目》总结。
- [RGMT 项目主页](https://zeonsunlightyu.github.io/RGMT.github.io/)
- [Any2Track 项目主页](https://zzk273.github.io/Any2Track/)

## 关联页面

- [Model-Based RL](./model-based-rl.md) — 世界模型在 Any2Track 中的辅助作用。
- [BeyondMimic](./beyondmimic.md) — 同样强调参考轨迹跟踪。
- [Switch](./switch-framework.md) — 引入了增强技能图与缓冲节点，彻底解决了 Any2Track 在处理大跨度动作切换时的失稳问题。
- [Domain Randomization](../concepts/domain-randomization.md) — 历史编码器本质上是在线识别被随机化的环境参数。
