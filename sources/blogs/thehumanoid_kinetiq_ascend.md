# KinetIQ Ascend: Towards 100% Manipulation Reliability and Superhuman Speed

> 来源归档（blog / Humanoid 官方）

- **标题：** KinetIQ Ascend: Towards 100% Manipulation Reliability and Superhuman Speed
- **类型：** blog
- **作者：** Humanoid Team
- **原始链接：** https://thehumanoid.ai/technology/kinetiq-ascend/
- **发表日期：** 2026-06（官方未标具体日；页面抓取于 2026-06-27）
- **入库日期：** 2026-06-27
- **抓取方式：** 官方博客页面直接抓取（WebFetch）
- **一句话说明：** Humanoid 发布 **KinetIQ Ascend**：在 **KinetIQ** 框架上为 **CFM 基座 VLA（System 1）** 增加 **真机 24/7 PPO 强化学习** 后训练栈，面向产线 **99.9% 成功率 @ 人类或超人类速度**；在 **Alpha 双臂人形** 上三项生产任务（供料、拣选交接、双手 tote 搬运）用 **数天 robot-time** 实现 **42%–100%+ 吞吐提升** 与 **10–20× 失败率下降**，并报告 **仅训瓶颈阶段/单物体仍泛化** 等意外发现。

## 核心摘录（归纳，非全文）

### 问题重框与目标

- **工业 KPI：** 工位机器人每天重复数千次；目标 **99.9% 任务成功率 @ 人类或超人类速度**，且可预测到足以规划部署。
- **端到端深度学习：** 传感器读数直接映射运动；核心架构为 **VLA（Vision-Language-Action）**。
- **BC 天花板：**
  1. **速度与质量受示教者上限约束**——单纯加速回放会因执行器动力学（夹爪闭合、关节加速度）导致时序失配；
  2. **被动模仿难达工业可靠性**——只学「做什么」不学「失败代价」；
  3. **因果混淆（causal confusion）**——操作者拥有策略不可见的记忆/算力/视角，模仿可能 latch 到虚假相关。
- **RL 角色：** 在真实物理动力学下 trial-and-error 优化显式奖励，可突破示教者速度与质量，并抑制虚假相关导致的失败长尾。
- **栈分层：** **KinetIQ System 0** 已用 RL 做全身控制；**KinetIQ Ascend** 将 **端到端视觉引导 RL** 上推到 **System 1 操作 VLA**。

### 真机 RL 基础设施

| 主题 | 要点 |
|------|------|
| **Sim vs Real** | 仿真用于方法与样本效率研究、数字孪生可达 100% 成功；**部署后持续学习**需求使 **真机 RL** 成为主路径（无 sim2real gap）。真机与仿真训练曲线 **高度相关**，作者据此声称对 100% 可靠性有 **line of sight**。 |
| **解耦训练/推理** | 数百台机器人 **机载推理（NVIDIA Jetson Thor）**；trainer 在云端，worker（真机或仿真）拉权重、采 rollout、回传 transition；可 **真机+仿真并行采样本**。 |
| **PPO + CFM** | 操作策略为 **条件流匹配（CFM）** BC 预训练；RL 需 action log-prob → **ODE→SDE 注入高斯噪声**（类 Flow-SDE）；实测 ReinFlow、FPO、Flow-GRPO、RAM 等在本规模/难度下不足，收敛到 **PPO + 单随机步加噪 + 末步无噪减抖动**。 |
| **Critic** | **回归式 critic**（CLS token + 线性头），policy/critic 权重分离、同 BC 初始化；标准 clipped PPO + GAE。 |
| **Prefix CFM loss** | 异步推理 **action prefix conditioning**；RL 只奖励执行段导致 prefix 复制崩溃、idle 臂漂移、训练发散 → 增加 **ℒ_prefix-CFM** 稳定 chunk 边界。 |
| **探索安全** | 主动柔顺 + **腕部力矩阈值自动终止** + 反向回放缩臂「痛觉反射」；安全终止进入 reward，RL 学会 **预防** 过大施力。 |
| **奖励** | **稀疏通用**：二元任务成功 + 可检测可恢复错误惩罚 + 折扣隐含时间压力；无任务专属 dense shaping。 |
| **非平稳环境** | 真机环境漂移（光照、操作员换班、磨损）→ **~10% episode 用冻结参考策略做在线 A/B baseline**；所有报告增益相对 **并发 baseline** 而非陈旧起点。 |
| **Sampler-trainer gap** | Thor 采样 vs Hopper/Blackwell 训练数值不一致 → trainer 侧 **重算 log-prob** 用于 importance ratio。 |
| **Speed curriculum** | 提高 policy **FPS**（sport mode）先掉质再 RL 恢复；须 **分步提速**（如 60→75→90 FPS），否则 reward 崩塌不可恢复。 |

### 三项生产任务结果（Alpha 双臂系统，BC 遥操作基线 → RL）

| 任务 | 训练重点 | Robot-time | 吞吐变化 | 成功率变化 | 备注 |
|------|----------|------------|----------|------------|------|
| **Machine feeding** | 仅 **拣选阶段** RL（传送易） | ~5 天 | **291→412 件/h（+42%）** | pick 0.60→0.67 | 速度课 60→75→90 FPS；**全任务提升来自瓶颈阶段训练** |
| **Object picking & handover** | 仅 **水瓶** RL | ~3 天 | **+85%**（水瓶） | **80%→98%**（~10× 失败↓） | 未见物体（罐 +40%、袋 +13%）；**臂条件指令** 未练仍保留 |
| **Bimanual tote handling** | 全任务同配方 | ~4 天 | **122→279 件/h（>2×）** | **77.6%→98.9%**（~20× 失败↓） | 与前三任务差异大但 **零任务专属改动** |

### 讨论与产品叙事

- **BC 角色重定义：** 有 RL 后 BC 只需覆盖 **行为模态**，不必一次训到可部署。
- **车队持续学习：** 部署后同一解耦 RL 环；**监督员介入 = 失败信号**。
- **RL 作数据引擎：** rollout 入库预训练下一代，**零 embodiment gap**。
- **自声称：** 据其所知，首次公开 **生产 VLA 上端到端视觉 RL**、**真机双臂人形**、**真实部署条件** 的组合演示。

## 对 wiki 的映射

- [kinetiq-ascend](../../wiki/entities/kinetiq-ascend.md)（方法/系统实体 + Mermaid 真机 RL 流水线）
- 交叉：[VLA](../../wiki/methods/vla.md)、[Behavior Cloning](../../wiki/methods/behavior-cloning.md)、[Manipulation](../../wiki/tasks/manipulation.md)、[Bimanual Manipulation](../../wiki/tasks/bimanual-manipulation.md)、[Action Chunking](../../wiki/methods/action-chunking.md)、[Sim2Real](../../wiki/concepts/sim2real.md)、[Curr-0](../../wiki/entities/current-robotics-curr0.md)、[Green-VLA](../../wiki/entities/paper-greenvla-staged-vla-humanoid.md)、[ROVE](../../wiki/entities/paper-rove-humanoid-vla-intervention.md)

## 可信度与使用边界

- **公司官方博客**，非 peer-reviewed；定量除上表外多为作者自报，需独立复现前不宜作硬基准。
- **硬件细节：** Alpha 双臂平台、Jetson Thor、具体 DoF/夹爪参数博客未完整公开。
- **「首次」声明** 为作者自评，学术界/产业界并行工作需持续对照。
- 图表与视频为定性+部分置信区间，**未提供** 完整开源代码或可复现 benchmark 包。

## Citation

```bibtex
@article{
    humanoid2026kinetiqascend,
    author = {Humanoid Team},
    title = {KinetIQ Ascend: Towards 100\% Manipulation Reliability and Superhuman Speed},
    journal = {Humanoid Blog},
    year = {2026},
    url = {https://thehumanoid.ai/technology/kinetiq-ascend/},
}
```
