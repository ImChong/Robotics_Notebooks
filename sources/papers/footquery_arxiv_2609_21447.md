# FootQuery: Future-Touchdown-Guided Retrieval from Depth History for Perceptive Humanoid Locomotion

> 来源归档（ingest）

- **标题：** FootQuery: Future-Touchdown-Guided Retrieval from Depth History for Perceptive Humanoid Locomotion
- **类型：** paper / humanoid / perceptive locomotion / depth history / touchdown prediction / cross-attention / PPO / sim2real
- **出处：** arXiv 预印本，2026-09（[2609.21447](https://arxiv.org/abs/2609.21447)）
- **论文链接：** <https://arxiv.org/abs/2609.21447>
- **PDF：** <https://arxiv.org/pdf/2609.21447>
- **作者：** Tao Dong *、Jia Yu、Yuxuan Fan、Linna Zhao、Jiaqi Gong、Andong Yang、Chao Gao、Guyue Zhou †（* 同等贡献；† 通讯）
- **机构：** 清华大学智能产业研究院（AIR）；清华大学电机工程与应用电子技术系；北京科技大学（USTB）；南洋理工大学（NTU）
- **项目页：** **无**（截至 2026-09-22：arXiv 与 HTML 版均未列独立项目页）
- **入库日期：** 2026-09-22
- **一句话说明：** 提出 **FootQuery**：从本体预测每只脚 **下一触地点分布**，以此 **查询稀疏采样的深度历史**；训练期将 **已实现接触** 投影回历史深度图监督 attention；配合渐进辅助力课程与楼梯 **踏面中线** 塑形，在 **Unitree G1** 上单策略完成户外楼梯与室内楼梯/平台/沟混合路线。

## 相关资料（策展）

| 类型 | 链接 | 说明 |
|------|------|------|
| 论文 | [arXiv:2609.21447](https://arxiv.org/abs/2609.21447) | 原文 |
| 相邻（落脚预测 + 奖励） | [SSR](../../wiki/entities/paper-ssr-humanoid-open-world-traversal.md) | 想象落脚用于 **训练奖励**；FootQuery 用于 **部署期检索** |
| 相邻（深度查询） | [CReF](../../wiki/entities/paper-cref.md) | 本体查询 **当前** 深度；FootQuery 查询 **历史** 深度 |
| 相邻（高程重建） | [SOLO](../../wiki/entities/paper-solo.md) | 逐格高程 QR；FootQuery 不做显式高程图 |
| 相邻（单深度重建） | [DPL](../../wiki/entities/paper-notebook-dpl-depth-only-perceptive-humanoid-locomotion-vi.md) | 交叉注意力重建局部高程 |
| 平台 | Unitree G1 | 12 腿关节控制；机载深度 + 本体历史 |

## 摘要级要点

- **问题：** 复杂地形落脚点在 **触地瞬间** 常已离开当前视野（自遮挡 / 相机 FOV）；仅看最新深度会丢失关键支撑信息。
- **FootQuery 核心：** 每只脚从本体预测 **下一触地点均值/方差** → 与 per-foot 特征组成 **query**，对 **288 个历史 depth token** 做 **4-head cross-attention**；**GRU** 汇总全局视觉记忆；与检索特征融合出动作。
- **训练监督：** 已实现触地点投影到历史深度帧，在 **当时可见 ROI** 上监督 read attention（相对 SSR 的奖励构造、SOLO 的高程 MSE，监督目标是 **历史图像区域**）。
- **训练辅助（仅训练期）：** **渐进辅助力课程**（按 pre-clamp 力需求与生存统计调节骨盆支撑后撤至 0）；**event-consistent tread-midline shaping**（摆动早期锁定目标踏面并奖励向中线推进 + 交替踏面接触）。
- **部署：** 仅 **本体 + 机载深度历史**；非对称 actor–critic + **PPO**。
- **仿真（最难档）：** 相对 ablation **NoFootQuery**，完整系统在 **20 cm 楼梯 / 50 cm 沟 / 50 cm 平台** 成功率分别 **87% / 94% / 91%**（+13 / +60 / +55 pp）；相对 **Humanoid Parkour Learning (HPL)** 在平台与楼梯上更高。
- **诊断（楼梯）：** 未来触地点在 **当前 ROI 可见** 仅 **7.82%**，在 **保留历史帧** 中可见 **67.89%**（其中 **60.07%** 仅历史可见）——直接动机检索机制。
- **真机：** **Unitree G1** 单策略：户外楼梯上行 + 室内楼梯上下/平台/沟连续路线。

## 核心摘录（面向 wiki 编译）

### 1) 与 SSR / CReF / SOLO 的分工

| 方法 | 视觉组织方式 | 落脚/接触信号用途 |
|------|--------------|-------------------|
| CReF | 本体查询 **当前** 深度 + GRU | 落脚候选奖励 |
| SSR | 想象落脚分布 | **构造训练奖励**（非部署检索） |
| SOLO | 逐格 QR 重建高程 | 地图单元高度 |
| **FootQuery** | **预测触地点 → 查询深度历史** | **部署期 per-foot 检索** + 历史 ROI 监督 |

**对 wiki 的映射：** [`wiki/entities/paper-footquery-perceptive-humanoid-locomotion.md`](../../wiki/entities/paper-footquery-perceptive-humanoid-locomotion.md)

### 2) 架构要点（Table I / Fig. 4）

- 浅层 CNN → 每帧 local tokens（288）+ frame-level → GRU → 全局视觉记忆 $m_t^{vis}$。
- Foot Encoder：$z_t^i = F([o_t, z_t^{prop}], e_i)$；Touchdown Head 输出对角高斯 $(\mu, \sigma)$（pelvis-yaw XY，有界）。
- Query：$Q_t^i = W_Q z_t^i + \phi(\mathrm{sg}[\mu/s_{xy}, \sigma/s_{xy}])$ → cross-attention → 64-d foot context。
- Actor 另收 foot contexts、detach 的 touchdown 与 state estimate。

**对 wiki 的映射：** 同上实体页「流程总览」Mermaid

### 3) 开源状态（步骤 2.5）

| 类别 | 状态 | 说明 |
|------|------|------|
| 官方项目页 | **无** | arXiv / HTML 无 `github.io` 或 Code 链接 |
| 官方代码 | **未开源** | 截至 2026-09-22 无公开仓库 |
| 预训练权重 | **未发布** | 论文未列模型下载 |

## 对 wiki 的映射

- 主沉淀：**[`wiki/entities/paper-footquery-perceptive-humanoid-locomotion.md`](../../wiki/entities/paper-footquery-perceptive-humanoid-locomotion.md)**
- 任务交叉：**[`wiki/tasks/stair-obstacle-perceptive-locomotion.md`](../../wiki/tasks/stair-obstacle-perceptive-locomotion.md)**、**[`wiki/tasks/humanoid-locomotion.md`](../../wiki/tasks/humanoid-locomotion.md)**
- 方法对照：**[`wiki/entities/paper-ssr-humanoid-open-world-traversal.md`](../../wiki/entities/paper-ssr-humanoid-open-world-traversal.md)**、**[`wiki/entities/paper-cref.md`](../../wiki/entities/paper-cref.md)**、**[`wiki/entities/paper-solo.md`](../../wiki/entities/paper-solo.md)**
