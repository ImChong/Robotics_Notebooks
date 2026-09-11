# 从域随机化到残差学习：Sim2Real 技术路线梳理

> 来源归档（blog / 微信公众号）

- **标题：** 从域随机化到残差学习：Sim2Real 技术路线梳理
- **类型：** blog
- **作者：** 自由度FreeDof（微信公众号）
- **原始链接：** https://mp.weixin.qq.com/s/K_6MibGXWwh9OL9eSZxOMg
- **发表日期：** 2026-08-23
- **入库日期：** 2026-09-11
- **抓取方式：** WebFetch（桌面 UA 返回微信验证页；正文由 WebFetch 可读通道获取）
- **原始抓取落盘：** [`sources/raw/wechat_freedof_sim2real_four_routes_dr_to_residual_2026-08-23.md`](../raw/wechat_freedof_sim2real_four_routes_dr_to_residual_2026-08-23.md)
- **一句话说明：** 以 **可辨识性** 为轴，把 Sim2Real 四条主流路线（系统辨识 / 域随机化 / 在线适应 / 残差学习）放在同一问题下比较；给出分层组合、症状查表与论文阅读判据。
- **开源状态（步骤 2.5）：** 综述文，无项目页、无代码仓 → **步骤 2.5 不适用**。

## 核心摘录（归纳，非全文）

### 核心问题

仿真里优化 $J_{\mathrm{sim}}(\pi)$，部署却要 $J_{\mathrm{real}}(\pi)$ 高——四条路线都在缩小这个差。共同问题：**真机动力学参数能否辨识？剩余误差如何处理？**

### 四条路线（立场对照）

| 路线 | 对可辨识性的立场 | 真机数据 | 典型代价 |
|------|------------------|----------|----------|
| **系统辨识** | 正面求解 | 专门辨识实验 | 参数须可辨、实验设计要求高 |
| **域随机化** | 放弃辨识，用覆盖换鲁棒 | 零（训练期） | 保守，敏捷上限受限 |
| **在线适应** | 推迟到运行时，只要求控制相关上下文可区分 | 部署期历史 | 激励不足时**静默退化**为 DR |
| **残差学习** | 不强求完整参数辨识，直接拟合修正项 | 真机 rollout | 只在训练分布内有效 |

### 分层组合（成熟系统读法）

先把能辨的辨出来（缩小不确定性）→ 对剩余不确定性做**窄 DR** → 再处理辨不出的部分（残差 / 适应）。顺序不能反。

### 文内点名代表作（→ 本库节点）

| 主题 | 代表 | wiki |
|------|------|------|
| 执行器 SysID + 零样本 | PACE | [paper-pace-sim2real-legged-robots](../../wiki/entities/paper-pace-sim2real-legged-robots.md) |
| 主动激励实验设计 | SPI-Active | （文内引用；本库暂无独立页） |
| 执行器残差 / UAN | Fey et al. 2025 | [actuator-network](../../wiki/methods/actuator-network.md) |
| 动作层残差 | ASAP | [paper-hrl-stack-25-asap](../../wiki/entities/paper-hrl-stack-25-asap.md) |
| 在线适应 | RMA / UP-OSI | [paper-rma-rapid-motor-adaptation](../../wiki/entities/paper-rma-rapid-motor-adaptation.md) |
| 多引擎 DR | PolySim | （文内引用） |
| 部署监控 | RAPT | （文内引用 arXiv:2602.01515） |
| 单关节实验设计深读 | 姊妹篇 | [sim2real-joint-sysid-experiment-design](../../wiki/methods/sim2real-joint-sysid-experiment-design.md) |

### 症状查表（节选）

| 症状 | 优先方向 |
|------|----------|
| 固定基座关节响应对不上 | 时间同步 / 单位 → 闭环执行器辨识（勿先调 PPO） |
| 回差、迟滞、柔性明显 | 可解释主效应 + 力矩层残差 |
| 固定基座吻合、落地失败 | 接触参数 / 状态估计 / 时延；需含接触辨识 |
| 敏捷动作跟不上 | 名义模型做准 + 动作层残差；步频是 reality gap 敏感代理 |
| 完全没有真机数据 | 宽 DR / ADR / 教师–学生 / 多引擎；仍需真机验收 |

## 对 wiki 的映射

- **新建对比页：** [sim2real-four-routes-identifiability](../../wiki/comparisons/sim2real-four-routes-identifiability.md)
- **姊妹篇（已入库）：** [wechat_freedof_sim2real_dynamics_identification.md](./wechat_freedof_sim2real_dynamics_identification.md) → [关节动力学辨识实验设计](../../wiki/methods/sim2real-joint-sysid-experiment-design.md)
- 交叉：[Sim2Real](../../wiki/concepts/sim2real.md)、[闭环误差分层工程](../../wiki/queries/sim2real-closed-loop-engineering.md)、[Sim2Real Approaches](../../wiki/comparisons/sim2real-approaches.md)

## 当前提炼状态

- [x] 公众号正文抓取与 raw 归档
- [x] 升格对比页（四条路线 + 可辨识性轴）
- [x] 与姊妹篇 SysID 实验设计页交叉链接
