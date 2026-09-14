# Behind RoboFinals: NVIDIA Isaac Lab-Arena and Lightwheel BenchHub

> 来源归档（site / 媒体文）

- **标题：** Behind RoboFinals: NVIDIA Isaac Lab – Arena and Lightwheel BenchHub
- **类型：** site（厂商技术解读）
- **来源：** 光轮科技（Lightwheel）
- **链接：** https://lightwheel.ai/media/robofinals-isaac-lab
- **入库日期：** 2026-09-14
- **一句话说明：** RoboFinals 双层基础设施：**NVIDIA Isaac Lab-Arena**（Scene/Embodiment/Task 解耦的开源评测框架，光轮×NVIDIA 联合开发）+ **Lightwheel BenchHub**（大规模 benchmark 托管与执行层，基于 Arena）。
- **代码：** [Isaac Lab-Arena](https://github.com/isaac-sim/IsaacLab-Arena) **已开源**；BenchHub 实现见 [LW-BenchHub](https://github.com/LightwheelAI/LW-BenchHub)
- **沉淀到 wiki：** [`wiki/entities/lightwheel-robofinals.md`](../../wiki/entities/lightwheel-robofinals.md)、[`wiki/entities/isaac-lab-arena.md`](../../wiki/entities/isaac-lab-arena.md)

---

## 官方要点摘录

### 评测基础设施叙事

- 前沿 VLA 已超越学术 benchmark；真机评测慢、贵、难复现 → **评测基础设施**成为瓶颈。
- 评测从「下游验证」变为指导 **数据采集、模型设计、学习本身** 的核心机制。

### 双层栈

```text
Lightwheel BenchHub（benchmark 托管 / 执行 / 规模化）
        ↓
NVIDIA Isaac Lab-Arena（Scene · Embodiment · Task 解耦评测核）
        ↓
Isaac Lab / 多物理后端（PhysX / Newton / …）
```

| 组件 | 角色 |
|------|------|
| **Isaac Lab-Arena** | 任务/机器人/场景解耦；系统化重组以测泛化；光轮扩展复杂任务逻辑与跨具身协议 |
| **BenchHub** | 托管、执行、扩展多 benchmark；域随机化；teleop + 确定性轨迹回放校准可行性/horizon/成功判据；**仅标准化 rollout 计分** |

### 已开源任务生态（文内数据，供对照 RoboFinals-100）

- **RoboCasa + LIBERO** 经 BenchHub 开源：原子技能到长时域复合；Gymnasium 注册。
- **场景：** RoboCasa 100 厨房 USD（10 layout × 10 style）；LIBERO 4 类桌面场景族。
- **具身：** 28 种机器人变体（G1、PandaOmron、Double Panda、SO100/101、Piper、ARX-X7s 等）。
- **受控 episode** 支持分布级评测。

## 对 wiki 的映射

| 主题 | 目标 wiki |
|------|-----------|
| RoboFinals 产品层 | [`wiki/entities/lightwheel-robofinals.md`](../../wiki/entities/lightwheel-robofinals.md) |
| Arena 开源框架 | [`wiki/entities/isaac-lab-arena.md`](../../wiki/entities/isaac-lab-arena.md) |
| BenchHub 工程样例 | [`wiki/entities/lw-benchhub-tour.md`](../../wiki/entities/lw-benchhub-tour.md) |
