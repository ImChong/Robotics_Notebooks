# Uni-VLaT: Whole-Body Tactile Adaptation of VLA Policies（arXiv:2609.35450）

> 来源归档（ingest）

- **标题：** Uni-VLaT: Whole-Body Tactile Adaptation of VLA Policies for Humanoid Loco-Manipulation
- **类型：** paper / vla / humanoid / tactile / loco-manipulation
- **arXiv abs：** <https://arxiv.org/abs/2609.35450>
- **PDF：** <https://arxiv.org/pdf/2609.35450>
- **项目页：** <https://uni-vlat.github.io/> — 归档见 [`sources/sites/uni-vlat-github-io.md`](../sites/uni-vlat-github-io.md)
- **代码：** **截至 2026-09-30 未列链接** — 项目页无 GitHub / HF；页内仍标 Anonymous Authors（双盲形态）
- **机构：** 清华大学（Tsinghua）、北京航空航天大学（Beihang）、中国传媒大学、香港大学（HKU）等 — Zihao Wang、Mengdi Xu 等
- **入库日期：** 2026-09-30
- **一句话说明：** 在 **冻结预训练 VLA**（Isaac-GR00T / π0.5）上增加 **全身分布式触觉通路** + **触觉锚定的多模态未来表征预测**（触觉/本体/视觉 latent）；Unitree G1 五任务均值 **75%** vs 无触觉 **32%**、仅触觉输入 **68%**。

## 相关资料（策展）

| 类型 | 链接 | 说明 |
|------|------|------|
| 项目页 | <https://uni-vlat.github.io/> | 五任务 demo、主表、跨 backbone 与消融 |
| 低层控制 | SONIC | 64-D motion token → 全身 joint PD；Protocol v4 |
| 对照 | WT-UMI、HTD、TACT | 全身触觉 / 预测监督谱系 |

## 摘要级要点

- **问题：** 人形 loco-manipulation 中接触决定响应，但视觉+本体对 **遮挡接触** 不敏感；稀疏 F/T 难覆盖 **空间分布触觉**。
- **方法：** 八区域 taxel → 区域 MLP → 8 空间 token → 4 帧因果时序 Transformer；**门控** 注入 DiT；**post-DiT 触觉池化** 为锚，三预测头监督 **未来 4 步** 触觉/本体/视觉 **绝对 latent**（flow-matching 动作损失并行）。
- **部署：** 去掉预测头与 target encoder；输出 **40×64** motion chunk 给 **SONIC** 解码。
- **数据：** 每任务 **50** 演示；主表 **20** rollouts/配置（π0.5 部分任务 10）。

## 核心摘录（面向 wiki 编译）

### 1) Isaac-GR00T 五任务（%）

| Task | No Tactile | Tactile w/o Pred. | Uni-VLaT |
|------|------------|-------------------|----------|
| Back-Tap Walking | 0 | 85 | 85 |
| Table Sweeping | 45 | 60 | **75** |
| Basket Loading | 30 | 65 | **80** |
| Human–Robot Hugging | 55 | 80 | 80 |
| Composed Cleanup | 30 | 50 | 55 |
| **Average** | **32** | **68** | **75** |

### 2) 跨 backbone（节选）

- Table Sweeping：GR00T **45→75**；π0.5 **30→60**
- Back-Tap Walking：GR00T **0→85**；π0.5 **0→90**

### 3) 消融（Isaac-GR00T，平均）

- Full Uni-VLaT **80%** vs Multimodal Prediction **60%** vs Pre-DiT 预测 **55%** vs Delta target **35%**

## 对 wiki 的映射

- 新建：[paper-uni-vlat](../../wiki/entities/paper-uni-vlat.md)
- 交叉：[vla](../../wiki/methods/vla.md)、[loco-manipulation](../../wiki/tasks/loco-manipulation.md)、[paper-loco-manip-07-wt-umi](../../wiki/entities/paper-loco-manip-07-wt-umi.md)、[unitree-g1](../../wiki/entities/unitree-g1.md)

## 当前提炼状态

- [x] arXiv + 项目页核查（无代码链）
- [ ] 待跟进：去匿名后 GitHub / 权重发布
