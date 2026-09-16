# ResSafe（arXiv:2609.15988）

> 来源归档（ingest）

- **标题：** ResSafe: Learning Safety Filtering with Residual Reinforcement Learning for Humanoids
- **简称：** ResSafe
- **类型：** paper / humanoid / safe-rl / residual-policy
- **arXiv：** <https://arxiv.org/abs/2609.15988>
- **PDF：** <https://arxiv.org/pdf/2609.15988>
- **项目页：** <https://sciautonomy.github.io/ResSafe_Web/> — 归档见 [`sources/sites/ressafe-sciautonomy.md`](../sites/ressafe-sciautonomy.md)
- **机构：** 加州大学伯克利分校（UC Berkeley）；加州大学洛杉矶分校（UCLA，部分作者）
- **入库日期：** 2026-09-16
- **一句话说明：** 名义策略只管任务性能，残差策略学安全修正，在 G1 极端平衡与随机载荷下实现更好的性能–安全帕累托前沿。

## 开源状态（步骤 2.5，2026-09-16）

| 组件 | 状态 |
|------|------|
| 项目页 Code 按钮 | **Code (Coming Soon)** — 截至入库日无 GitHub URL |
| 论文 PDF | 可获取 |

**结论：待发布** — 项目页已挂入口但尚未放出仓库；真机视频与仿真结果可复核论文/站点。

## 核心摘录

### 摘录 1：核心思想

- 人形 RL 策略仍可能输出导致失稳或跌倒的不安全动作。
- **ResSafe** 用 **残差强化学习** 作 **隐式安全过滤器**：名义策略专注任务性能，残差策略学安全修正，避免在单一策略里精细调 competing reward terms。
- 相对学习型安全过滤基线与名义参考策略，在挑战性人形平衡任务上 **更安全且仍保持可用性能**。

**对 wiki 的映射：** [paper-ressafe](../../wiki/entities/paper-ressafe.md)

### 摘录 2：G1 设定

- **Unitree G1**：状态维 **n=58**，动作维 **m=29**；仿真用 **Isaac Gym**。
- 任务：极端平衡、随机载荷扰动；仿真 + 真机硬件验证；可泛化到不同 reference policy checkpoint。

**对 wiki 的映射：** 同上

### 摘录 3：与显式安全过滤的对比（项目页摘要）

- 学习型安全过滤若只编码 **半空间约束** 而非安全动作本身，泛化会受限；ResSafe 通过残差 decoupling 改善 performance–safety–robustness 权衡。

**对 wiki 的映射：** 同上（与其他工作对比 / 局限）

## 当前提炼状态

- [x] 项目页 Code 状态核查（Coming Soon，2026-09-16）
- [x] wiki 映射：`wiki/entities/paper-ressafe.md`
