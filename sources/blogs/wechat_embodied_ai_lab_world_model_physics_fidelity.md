# 看完最近这批机器人世界模型论文，我想弄清楚模型到底学到了多少真实物理

> 来源归档（blog / 微信公众号）

- **标题：** 看完最近这批机器人世界模型论文，我想弄清楚模型到底学到了多少真实物理
- **类型：** blog
- **作者：** 具身智能研究室（微信公众号）
- **原始链接：** https://mp.weixin.qq.com/s/OawDKruG8zEepiy-x1nKuA
- **发表日期：** 2026-07-27
- **入库日期：** 2026-07-27
- **抓取方式：** [Agent Reach](https://github.com/Panniantong/Agent-Reach) v1.5.0 + `wechat-article-for-ai`（Camoufox）；`--no-images`；Jina Reader 对 `mp.weixin.qq.com` 返回 CAPTCHA，未采用
- **原始抓取：** [`sources/raw/wechat_world_model_physics_fidelity_2026-07-27/`](../raw/wechat_world_model_physics_fidelity_2026-07-27/)
- **一句话说明：** 前篇把运动控制接进 WAM 的五种位置后，本篇改按 **「动作执行后模型用什么记录世界变化」** 读读：潜变量 → 视频 → 持续状态 → 物理相关信号 → 回控结果；并给出动作/动力学敏感性、可执行性、策略相关性四类测试优先序。

## 核心摘录（归纳，非全文）

### 总问题

- 「世界模型」这个标签已经不够用：Ego-VCP、MotionWAM、RynnWorld-4D、VT-WAM 都叫 WM，但内部保存/预测的东西差得很远。
- 阅读顺序建议：**先弄清预测头输出，再看网络结构。**

### 输出阅读轴（文内 taxonomy，非严格互斥）

| 输出族 | 代表工作 | 读法 |
|--------|----------|------|
| **低维潜在状态** | World Models、PlaNet、DreamerV3、TD-MPC2 | 编码→动作→滚动 latent→估计回报→选动作；快，但压缩可能丢掉接触/动量细节 |
| **未来图像/视频** | UniSim、IRASim、V-JEPA 2（潜空间规划中间路线） | 画面可检查；风险是「画面连续 ≠ 动力学」 |
| **持续状态** | WorldWeaver | 共享场景/智能体/镜头外信息用寄存器跨片段读写 |
| **动作 vs 世界效应分解** | DWM（Separating World Effects，arXiv:2607.18715） | 把转移拆成动作诱导与动作无关世界分支；自主动态环境 CEM +13.1pp |
| **视觉动作接口** | Masked Visual Actions | 部分揭示像素轨迹作动作；~15h 微调；RoboCasa 策略评估 r=0.982 |
| **几何/运动信号** | RynnWorld-4D、MECo-WAM | RGB+深度+光流；训练期 4D 专家、部署撤掉 |
| **触觉** | VT-WAM | 视觉、触觉形变、动作联合生成；同步与跨传感器难 |
| **物理混合** | PhysCoRe | 可微 MPM + 材料参数估计 + 残差修正可变形体 |
| **评测诊断** | Imagined Rollouts are Kinematic Not Dynamic、KineBench、Thinking in Video | 运动学幻觉、IDM-free 可执行性、感知–预测差距 |

### 作者强调的四类测试优先序

1. **动作敏感性** — 方向/幅度/时机变了，预测是否跟着变
2. **动力学敏感性** — 质量/摩擦/外力/材料变化是否进入 rollout
3. **可执行性** — 轨迹放回机器人或高保真仿真能否跑
4. **策略相关性** — 模型排序与真机测试是否对齐

### 文内入口（策展口径）

| 工作 | arXiv / 入口 |
|------|----------------|
| World Models | <https://arxiv.org/abs/1803.10122> |
| PlaNet | <https://arxiv.org/abs/1811.04551> |
| DreamerV3 | <https://arxiv.org/abs/2301.04104> |
| TD-MPC2 | <https://arxiv.org/abs/2310.16828> |
| UniSim | <https://arxiv.org/abs/2310.06114> |
| IRASim | <https://arxiv.org/abs/2406.14540> |
| V-JEPA 2 | <https://arxiv.org/abs/2506.09985> |
| WorldWeaver | <https://arxiv.org/abs/2607.21594> · <https://vail-ucla.github.io/worldweaver/> |
| DWM（Separating World Effects） | <https://arxiv.org/abs/2607.18715> |
| Masked Visual Actions | <https://arxiv.org/abs/2607.19343> · <https://masked-visual-actions.github.io/> |
| RynnWorld-4D | <https://arxiv.org/abs/2607.06559> |
| MECo-WAM | <https://arxiv.org/abs/2607.05468> |
| VT-WAM | <https://arxiv.org/abs/2607.02503> |
| PhysCoRe | <https://arxiv.org/abs/2607.20653> |
| Imagined Rollouts… | <https://arxiv.org/abs/2607.05966> |
| KineBench | <https://arxiv.org/abs/2607.19876> · <https://github.com/minecraft-zzz/KineBench> |
| Thinking in Video | <https://arxiv.org/abs/2607.17523> · <https://github.com/BRZ911/Thinking-in-Video> |

> **命名注意：** 文中 **DWM（Separating World Effects，arXiv:2607.18715）** 与本库已有 [Dexterous World Models（DWM，arXiv:2512.17907）](../../wiki/methods/dwm.md) **不是同一篇**；入库时拆成独立节点，勿合并。

## 对 wiki 的映射

- 主升格：[world-model-physics-fidelity-outputs](../../wiki/overview/world-model-physics-fidelity-outputs.md)
- **复用已有 complete 节点（不新建重复）：**
  - [DreamerV3](../../wiki/entities/paper-shenlan-wm-13-dreamerv3.md)
  - [Masked Visual Actions](../../wiki/entities/paper-masked-visual-actions.md)
  - [RynnWorld-4D](../../wiki/entities/paper-rynnworld-4d-rgb-depth-flow.md)
  - [MECo-WAM](../../wiki/entities/paper-meco-wam-4d-geometry-cotraining.md)
  - [VT-WAM](../../wiki/entities/paper-vt-wam-visuotactile-contact-rich.md)
  - [Ego-VCP](../../wiki/entities/paper-hrl-stack-33-ego_vision_world_model_for_humanoid.md)（文首对照）
  - [MotionWAM](../../wiki/entities/paper-motionwam-humanoid-loco-manipulation-wam.md)（文首对照）
  - [Dexterous DWM](../../wiki/methods/dwm.md)（仅作命名消歧，非本文 DWM）
- **新建独立论文实体：** World Models、PlaNet、TD-MPC2、UniSim、IRASim、V-JEPA 2、WorldWeaver、DWM-Separating、PhysCoRe、Imagined Rollouts、KineBench、Thinking in Video
- 相关：[WAM × 运动控制五路径](../../wiki/overview/wam-motion-control-five-paths.md)、[robot-world-models-training-loop-taxonomy](../../wiki/overview/robot-world-models-training-loop-taxonomy.md)、[Generative World Models](../../wiki/methods/generative-world-models.md)

## 可信度与使用边界

- 本文为 **微信公众号策展判断文**；taxonomy 与读法以本文为准；方法数字、开源状态与评测以各论文 PDF / 项目页 / 代码仓为准。
- 不把公众号作为唯一一手来源；每篇论文实体页的「参考来源」同时挂 arXiv / 项目页 / 代码归档。

## 当前提炼状态

- [x] 正文抓取与输出轴归纳
- [x] 文内入口与 wiki 映射（含去重 / 命名消歧）
- [x] 升格 overview + 补齐缺失论文实体（非 stub）
