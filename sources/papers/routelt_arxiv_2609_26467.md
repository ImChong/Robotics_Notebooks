# RouteRLT: VLA 与 RL 专家自动路由（arXiv:2609.26467）

> 来源归档（ingest）

- **标题：** RouteRLT: Learning When and Which RL Specialist Should Control a Vision-Language-Action Policy
- **类型：** paper / vla / rl / routing / manipulation
- **arXiv abs：** <https://arxiv.org/abs/2609.26467>
- **PDF：** <https://arxiv.org/pdf/2609.26467>
- **项目页：** 无独立项目页（IROS 2026 IARL workshop）
- **代码：** **未列链接**（2026-09-24）
- **机构：** 多伦多大学（University of Toronto）
- **入库日期：** 2026-09-24
- **一句话说明：** 冻结 SmolVLA 泛化策略 + 多枚 phase-specific RL specialist；phase selector 从 VLA 内部 latent 预测 controller ownership；stabilizer 抑切换抖动；action-boundary manager 在中途换控时作废 chunk 后缀。

## 核心摘录

### 1) 模块

- **Phase selector：** 因果 phase classifier → controller posterior $\mathbf{p}_t$。
- **Router stabilizer：** 抑制瞬态切换。
- **Action-boundary manager：** ownership 变化时立即生效，而非等下一 replan 边界。

### 2) headline 数字

| 设定 | Base VLA | RouteRLT |
|------|----------|----------|
| LIBERO 多物体 pick-and-place 全任务 SR | 85.00% | **92.22%** |
| 真机线缆 pickup+insertion 全轨迹成功 | 6.7% | **35.0%** |

- 仿真匹配 **privileged phase boundary** 路由，部署 **无** privileged 信号。

### 3) 开源状态

- arXiv + workshop 接受；**无** 公开代码 URL → **未列链接 / 待核实**。

## 对 wiki 的映射

- 新建：[paper-routelt](../../wiki/entities/paper-routelt.md)
- 交叉：[vla](../../wiki/methods/vla.md)、[manipulation](../../wiki/tasks/manipulation.md)、[smolvla 相关实体若存在]
