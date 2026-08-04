# ActFovea: Runtime Safeguarding for VLA Policies via Spatiotemporal Visual-Action Consistency（arXiv:2607.29169）

> 来源归档（ingest）

- **标题：** ActFovea: Runtime Safeguarding for VLA Policies via Spatiotemporal Visual-Action Consistency
- **缩写 / 框架：** **ActFovea**（Action-conditioned Foveation）
- **类型：** paper / vla / runtime-safety / anomaly-detection / plug-and-play / libero
- **arXiv：** <https://arxiv.org/abs/2607.29169>（v1，Submitted 2026-07-31，cs.RO，CC BY 4.0；PDF：<https://arxiv.org/pdf/2607.29169>）
- **代码（论文 comments 声明）：** <https://github.com/SunnyYWD/ActFovea>（Apache-2.0 + Gemma 条款）— 归档见 [`sources/repos/actfovea.md`](../repos/actfovea.md)
- **作者：** Wenda Yu¹、Tianshi Wang¹、Fengling Li²、Xin Li³、Jingjing Li⁴、Lei Zhu¹
- **机构：** 预印本仅标注上标编号 1–4，**未在正文/首页写出单位名称**；因此本次不写机构 tag（待后续版本或期刊版补）
- **入库日期：** 2026-08-04
- **一句话说明：** 不重训、不改 VLA 权重，在「观测→动作」接口上加一层运行时防护：用机器人运动学 + 本体状态 + 近期动作构造 **动作条件中央凹（foveation）** 区域，靠 **时空视觉–动作一致性** 检测扰动，可恢复的走「候选观测 → 动作块验证 → 受限执行」，不可恢复（冻结重放）的走 **有界安全失败**。

## 开源状态（步骤 2.5）

- **核查（2026-08-04）：** 论文 comments 直接给出 GitHub URL，仓库为 **openpi 的分叉改造**，含 `src/openpi/defense/`（威胁检测与恢复逻辑）、`src/openpi/policies/defense_policy.py`（运行时策略包装器）、`examples/libero/main.py`（评测 harness）、`scripts/serve_policy.py`（基线 / ActFovea 服务端）、`scripts/eval_libero_task_matrix.py`（四条件评测）、pytest 单测。
- **结论：** **已开源**（完整实现，无占位；Apache-2.0 主许可 + Gemma 许可条款）。**权重复用 π₀ 官方 checkpoint**（`OPENPI_CHECKPOINT_DIR`），仓库本身不发布新训练权重——这与「training-free」定位一致。
- **无独立项目页**（无 `*.github.io`），代码即唯一复现入口。

## 摘录 1：问题设定 —— 四类破坏「时空视觉–动作一致性」的运行时扰动

VLA 策略在干净条件下表现好，但部署期扰动会**打断视觉观测、机器人状态与已执行动作三者的时序对齐**。论文把这些扰动统一刻画为一致性违背，并分四类：

| 扰动 | 注入方式（LIBERO 实验） | 性质 |
|------|------------------------|------|
| **Spatial visual corruption**（视觉遮挡） | 持续性局部棋盘格叠加，alpha 0.5，短暂干净预热后开始 | 可恢复（空间） |
| **Temporal misalignment**（视觉延迟） | 多视角图像滞后 **3 帧**，与当前本体状态错配 | 可恢复（时间） |
| **Action-trajectory drift**（动作漂移） | 执行前按任务族施加相位条件窗口内的平滑运动扰动 | 可恢复（动作侧） |
| **Frozen-observation replay**（观测冻结重放） | 重复同一帧直至 episode 结束，模拟传感器锁死 | **不可恢复**，只能安全失败 |

**对 wiki 的映射：** 升格 [`wiki/entities/paper-actfovea.md`](../../wiki/entities/paper-actfovea.md)；与 [safety-filter](../../wiki/concepts/safety-filter.md)、[robot-safety-state-machine](../../wiki/concepts/robot-safety-state-machine.md)、[vla-deployment-guide](../../wiki/queries/vla-deployment-guide.md) 互链。

## 摘录 2：四个组件（§方法）

1. **Action-conditioned foveation（动作条件中央凹）**
   保留掩码 \(M^v_t=\mathrm{Dilate}(M^v_{c,t}\vee M^v_{\Gamma,t},r_m)\)：\(M_c\) 是投影夹爪接触点为心、半径 \(r_c\) 的圆盘；\(M_\Gamma\) 是沿预测轨迹路点、半径 \(r_\Gamma\) 的**运动走廊**。背景做归一化 / 平滑 / 去饱和的**有界弱化**（强度 \(\alpha\)），保留区保持原图保真：
   \(\tilde I^v_t=(1-\alpha \bar B^v_t)\odot I^v_t+\alpha \bar B^v_t\odot \mathcal E(I^v_t)\)。
   关键点：掩码**跟着预期交互移动**，而不是固定在图像坐标里。
2. **Consistency monitor（一致性监控）**
   风险分 \(R_t=\mathrm{clip}(\beta\bar r_t+(1-\beta)r_t+p^{cam}_t+p^{lag}_t+p^{cal}_t,0,1)\)。分量含：几何一致性（观测接触中心 vs 投影接触中心距离）、动态一致性（预测像移 vs 观测像移的方向与幅度）、时间证据（时间戳健康度、应有却缺失的局部运动、短历史匹配估计滞后、全局重放相似度）、动作–本体一致性；再加相机不可用 / 标定不一致 / 估计延迟三类惩罚。确定性 router 据证据模式与持续性把观测**路由到 delay / drift / replay 三类威胁**。
3. **Candidate bank + action-chunk verification（候选库与动作块验证）**
   候选包含：原始观测、foveated 观测、时间稳定候选（对付延迟）；对确认的局部叠加另做**空间修复** \(\hat X^v_t=\mathrm{clip}((P^v_t-\hat\alpha^v_t\hat Q^v_t)/(1-\hat\alpha^v_t),0,255)\)（先用稠密光流中值对齐到历史干净帧，再估计叠加图案与混合系数）。每个候选喂给**冻结的 VLA**，对返回的动作块打分 \(V_k=\mathrm{clip}(w^\top u_k+b_k,0,1)\)，\(u_k\) 聚合首动作方向、终点方向、运动幅度、平滑度、horizon、chunk 漂移等；\(b_k\) 是威胁条件加成。
4. **Risk-adaptive execution（风险自适应执行）**
   两级仲裁：\(\hat a^{mot}_{t,i}=\lambda^{mon}_t\lambda^{ver}_t a^{\star,mot}_{t,i},\ i<h_t=\min(h^{mon}_t,h^{ver}_t)\) —— 只缩放**运动维度**，夹爪指令保持；手段是运动阻尼、短 horizon 执行与 servo recovery。

**对 wiki 的映射：** 实体页画「监控 → 路由 → 候选 → 验证 → 有界执行 / 安全失败」流程图与源码运行时序图。

## 摘录 3：安全失败（不可恢复分支）

- **hold latch**：重放证据在多次恢复尝试后仍持续 → 闭锁，**停止查询策略**并抑制运动。
- **有界动作**：至多前置 1 个截断的反向动作，其余动作块填零运动 hold。
- **定位**：目标是**保守收敛**而非完成任务，显式转入终止性 safe-failure 状态，而不是无限期悬停。
- **效果**：检测后累计运动范数相对无防护基线降低 **99.87%**；检测后动作数 **259.2 → 2.0**（−99.23%）。

## 摘录 4：实验与结果（LIBERO ×4 suites，冻结 π₀）

- **规模：** LIBERO Spatial / Object / Goal / 10-task 共 **40 任务**，每任务 50 episodes → 每个「方法×场景」组合 **2000 episodes**。
- **指标：** SR；Gain（绝对百分点）；**NRR**（Normalized Recovery Rate）\(=\frac{SD+AF-SD}{SC-SD}\times100\%\)，即「相对干净性能的差距回收比例」。

| 扰动 | 干净基线 | 受扰基线 | +ActFovea | Gain | NRR |
|------|---------|---------|-----------|------|-----|
| Action drift | 92.7% | 83.1% | 90.1% | **+7.0 pp** | 73.1% |
| Visual delay | 92.6% | 76.2% | 86.0% | **+9.8 pp** | 59.8% |
| Visual overlay | 93.0% | 49.3% | 90.3% | **+41.0 pp** | **93.7%** |

**与其他免训练运行时手段对比：**

| 方法 | 无扰动 | Drift | Delay | Overlay |
|------|-------|-------|-------|---------|
| Base VLA（π₀） | 93.0% | 83.1% | 76.2% | 49.3% |
| Action Clip / Smoothing | 82.2% | 70.4% | 70.2% | 30.9% |
| Fixed Short Horizon | 91.7% | **89.9%** | 70.7% | 32.4% |
| Timestamp-Only Hold | 93.1% | 84.9% | **0.0%** | 48.5% |
| **ActFovea** | **93.8%** | **90.1%** | **86.0%** | **90.3%** |

读点：固定短 horizon 只在动作漂移上有效、在视觉侧反而更差；纯时间戳 hold 遇到「时间戳看着新、内容其实滞后」的延迟场景会**全盘卡死到 0%**；固定裁剪 / 平滑无扰动时就先掉 10.8 pp。

**冻结重放（2000 episodes）：**

| 方法 | 任务成功 | 及时安全失败 | 无防护失败 |
|------|---------|-------------|-----------|
| Base VLA | 3.05% | 0.00% | 96.95% |
| Timestamp-Only Hold | 0.00% | 0.00% | 100.00% |
| w/o Hold/Safe-Fail | 0.65% | 0.00% | 99.35% |
| **ActFovea** | 0.00% | **100.00%** | **0.00%** |

**消融（相对干净的 Gain）：**

| 去掉的组件 | Drift | Delay | Overlay |
|-----------|-------|-------|---------|
| w/o Threat Typing | +4.4 | +4.3 | **−7.6** |
| w/o Recovery Bank | +4.4 | +7.8 | **−33.3** |
| w/o Candidate Expansion | +7.5 | +9.2 | **−31.7** |
| w/o Action Verification | **−1.2** | +2.3 | +42.8 |
| Full ActFovea | +7.0 | +9.8 | +41.0 |

读点：**空间恢复靠「定位损坏区 + 造候选」**（去掉威胁分型 / 恢复库 / 候选扩展，overlay 直接由正转负）；**时间与动作侧恢复靠动作块验证**这道共享保守闸门（去掉后 drift 变 −1.2 pp）。overlay 在去掉验证后反而 +42.8，说明验证是**为一致性收紧、牺牲个别场景峰值**的通用闸门。

**对 wiki 的映射：** 实体页写「四类扰动 → 该配哪种防护」的选型表与「不要只看时间戳」的误区。

## 摘录 5：局限（论文自述）

1. **威胁模型边界：** 只覆盖「观测→动作接口之前」的扰动；执行器故障等接口之后的问题不在范围内。依赖**带时间戳的观测**与本体测量。
2. **无形式化避碰保证：** 安全失败是保守运动抑制，不是几何安全性证明。
3. **超参较多：** \(r_c,r_\Gamma,r_m,\alpha,\beta\)、分量权重与各类阈值均为固定实现常数，论文未给敏感性分析。
4. **恢复边界：** 持续重放**按设计就不可恢复**；有限视觉延迟与执行前动作漂移可处理，执行后的执行器异常不行。
5. **「training-free」不等于「零配置」：** 仍需运动学模型、相机标定与任务相关的动作包络约束。

**对 wiki 的映射：** 局限与风险节直接落这 5 条，并强调「运行时防护 ≠ 安全性证明」。

## 建议 wiki 动作

- 新建 **`wiki/entities/paper-actfovea.md`**（含流程总览 + 源码运行时序图 + 结论）。
- 新建 **`sources/repos/actfovea.md`**。
- 交叉：[`wiki/concepts/safety-filter.md`](../../wiki/concepts/safety-filter.md)、[`wiki/concepts/robot-safety-state-machine.md`](../../wiki/concepts/robot-safety-state-machine.md)、[`wiki/queries/vla-deployment-guide.md`](../../wiki/queries/vla-deployment-guide.md)、[`wiki/entities/libero-benchmark.md`](../../wiki/entities/libero-benchmark.md)、[`wiki/entities/paper-pi05-open-world-vla.md`](../../wiki/entities/paper-pi05-open-world-vla.md)。
