# Curr-0: A Loco-Dexterous Manipulation Model for Humanoids

> 来源归档（blog / Current Robotics 官方）

- **标题：** Curr-0: A Loco-Dexterous Manipulation Model for Humanoids
- **类型：** blog
- **作者：** Current Robotics Team
- **原始链接：** https://current-robotics.com/blog/curr-0
- **发表日期：** 2026-06
- **入库日期：** 2026-06-16
- **抓取方式：** 官方博客页面直接抓取（WebFetch）
- **一句话说明：** Current Robotics 发布 **Curr-0** 人形 **loco-dexterous manipulation** 全栈系统：自研可穿戴 **HumanEx** 将自然人体行为转为可重定向训练数据（**21,000 h** 人类数据、**2,800 h** 全身演示）；**三系统架构**（System 2 推理接地 / System 1 全身运动与平衡 / System 0 灵巧手物交互）在 **70+ DoF** 人形上 **端到端单策略** 闭环执行；配套 **多模态交互世界模型** 作可扩展评测与 **Human-in-the-World-Model** 部署后训练。

## 核心摘录（归纳，非全文）

### 问题重框

- **loco-dexterity 不可分解：** 站姿决定抓取、躯干决定可达、足端接触决定平衡、手–物接触改变全身未来运动；真实任务（撕茶包、点香、盖章、踩踏板倒垃圾、抱玩偶过门）要求 **移动、灵巧接触、工具使用与恢复** 在同一闭环策略中 **持续耦合**，而非「先走再手」流水线。
- **系统缺口：** 缺少可规模化的 **野外高质量全身–灵巧–感知–接触** 人类数据，以及面向长时程、难重置、安全敏感人形任务的 **评测与迭代环**。

### HumanEx：人→人形数据缩放

| 维度 | 要求 |
|------|------|
| Embodied | 保留感知–身体–手–物体–任务进度的物理结构 |
| Interactive | 闭环响应世界变化，而非静态姿态序列 |
| Retargetable | 可校准、同步、重定向到目标人形运动学与控制约束 |
| Scalable | 随人类日常活动扩展，而非仅随机器人机队小时数 |

- **HumanEx** 软可穿戴栈：全身本体、绑带 **EMG**、环境运动感知；按任务可配置从轻量 egocentric 视频到全身 embodied 采集。
- **三轴评价 wearable：** Action Fidelity / Action Diversity / Visual Diversity。
- **incidental human behavior：** 自然任务中涌现、难以用语言指令或脚本演示覆盖的物理先验。
- **缩放律叙事：** 从 **robot-hour** 转向 **human-task-hour**。

### Curr-0：三系统架构

| 层级 | 角色 |
|------|------|
| **System 2** | 语言 + 视觉 + 机器人状态 → 任务条件 **latent**、子任务推理、未来视觉–运动预测；对上身姿态与手臂–躯干协调有 **辅助监督** |
| **System 1** | latent → **行走、躯干、姿态、平衡、手臂可达** 的全身稳定行为 |
| **System 0** | **手–物接触** 专责：抓、捏、调整、再抓、工具作用；与 System 1 协调；**21-DoF Wuji Hand** |

- **训练混合：** System 1 先在全身数据上预训练；System 2 在手部中心数据上预训练；再在完整人类数据混合上 **联合微调** 为 **单策略共享权重**。
- **演示任务（博客视频叙事）：** 泡茶（双手机动撕袋、抑制吊牌摆动）、文件盖章（双手协作开盖）、点香（细棒对齐 + 打火机）、踩踏板倒垃圾（单脚保持踏板 + 多物分持释放）、玩偶过门（肘推门 + 跪放篮中）。

### 交互多模态世界模型

- **可扩展评测器：** 闭环 rollout + 自动打分；博客称与真机成功率 **高度相关**，优于传统 sim-to-real 基线对齐度。
- **多模态条件：** 除 RGB 外需 **本体、力、触觉** 等，以捕捉足端不稳、滑移、负载下平衡恢复等 RGB 难判失败。
- **Human-in-the-World-Model：** 策略在世界模型中 rollout，人类在 **错误/不确定/失败倾向** 时分支介入；纠正片段用于规模化 **post-training**；真机仍用于 grounding 与终验。

### 公司定位（博客自述）

- **Current Robotics** 全栈叙事：**真实世界数据 + 可扩展模型设计 + 快速迭代环**；Logo 灵感来自达·芬奇 **Vitruvian Man**。

## 对 wiki 的映射

- [current-robotics-curr0](../../wiki/entities/current-robotics-curr0.md)（系统实体 + Mermaid 全栈图）
- 交叉：[Loco-Manipulation](../../wiki/tasks/loco-manipulation.md)、[舞肌科技 Wuji Hand](../../wiki/entities/wuji-robotics.md)、[VLA](../../wiki/methods/vla.md)、[MotionWAM](../../wiki/entities/paper-motionwam-humanoid-loco-manipulation-wam.md)、[LEGS](../../wiki/entities/paper-legs-embodied-gaussian-splatting-vla.md)、[生成式世界模型](../../wiki/methods/generative-world-models.md)、[World Action Models](../../wiki/concepts/world-action-models.md)、[运动重定向管线](../../wiki/concepts/motion-retargeting-pipeline.md)

## 可信度与使用边界

- 本文为 **公司官方博客**，非同行评审论文；**定量指标**（除数据小时数外）多为定性视频演示与相关性叙事，**独立复现前不宜作硬基准引用**。
- **硬件栈**（具体人形平台型号、除 Wuji Hand 外的本体）博客未完整公开 datasheet 级参数。
- **世界模型与真机相关性** 为作者自报，需对照后续技术报告或论文。

## Citation

```bibtex
@article{
    currentrobotics2026curr0,
    author = {Current Robotics Team},
    title = {Curr-0: A Loco-Dexterous Manipulation Model for Humanoids},
    journal = {Current Robotics Blog},
    month = {June},
    year = {2026},
    url = {https://current-robotics.com/blog/curr-0},
}
```
