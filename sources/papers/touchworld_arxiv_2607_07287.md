# TouchWorld：Predictive and Reactive Tactile Foundation Model for Dexterous Manipulation

> 来源归档（ingest）

- **标题：** TouchWorld: A Predictive and Reactive Tactile Foundation Model for Dexterous Manipulation
- **类型：** paper / manipulation / tactile-sensing / vla / world-model / dexterous-manipulation / hierarchical-policy
- **arXiv abs：** <https://arxiv.org/abs/2607.07287>
- **arXiv HTML：** <https://arxiv.org/html/2607.07287v2>
- **PDF：** <https://arxiv.org/pdf/2607.07287>
- **项目页：** <https://phanes-lab.github.io/TouchWorld-website/>
- **机构：** 哈尔滨工业大学（深圳）（Harbin Institute of Technology, Shenzhen）；PHANES AI
- **通讯作者：** Shuo Yang（shuoyang@hit.edu.cn）
- **提交 / 修订：** 2026-07-08 提交、2026-07-09 修订（v2）
- **入库日期：** 2026-07-16
- **一句话说明：** **预测–反应式触觉基础模型**：四层多时钟层级（子任务规划 + 触觉世界模型预测子目标 → 视触觉目标条件 VLA 名义动作块 → TRT 高频触觉残差精修），在 **6 项长程接触丰富灵巧真机任务** 上干净设置 **65.0%**、人为扰动 **53.7%** 平均成功率，较最强基线 **FTP-1** 分别 **+15.7 / +18.5 pt**。

## 摘要级要点

- **问题：** 日常灵巧操作既要 **预见接触应如何演化**，也要在滑移、错位、抓不稳、力不匹配时 **毫秒级纠偏**；多数 VLA 把触觉当低频附加 token，与慢语义推理、动作块生成、快触觉反馈 **挤在同一单体环路**。
- **核心范式：** 触觉同时作 **预测接触参考**（Tactile World Model 生成视触觉子目标）与 **快反馈信号**（Tactile Residual Transformer 在线残差修正）。
- **四层架构：** (1) **Subtask Planner**（Qwen3-VL-4B + 记忆）；(2) **Tactile World Model**（Wan2.2-TI2V-5B，EgoTouch 人视频预训练 + 10 h 机端微调）；(3) **Visuo-Tactile Goal-Conditioned Policy**（扩散 Transformer + flow matching，触觉统一图像表征）；(4) **TRT 触觉残差层**（$W=16$ 滑动窗口、每 $C=4$ 步提交）。
- **硬件：** 人形 + **Wuji** 灵巧手 + **JQ-Industries** 触觉手套；遥操 **Meta Quest + Touch Plus + Wuji Glove**。
- **6 任务基准：** Water Flower、Tabletop Clearing、Cup Insertion、Power Plug Insertion、Pot Wiping、Tissue Pulling；每任务 **200** 条遥操训练轨迹、**100** 次评测 rollout；含 **人为扰动** 设置。
- **主结果（Table 1）：** TouchWorld **65.0% / 53.7%**（干净 / 扰动）vs FTP-1 **49.3% / 35.2%**、π₀.₅ **40.7% / 27.7%**、GR00T N1.7 **39.3% / 26.0%**。
- **开源状态（项目页核查，2026-07-16）：** 项目页 **仅列 Paper（arXiv PDF）**；**未列 GitHub、Hugging Face、Zenodo 或数据集下载链接**。

## 核心论文摘录（MVP）

### 1) 多时钟层级：把语义规划、触觉预测、名义动作与快反馈解耦

- **链接：** <https://arxiv.org/html/2607.07287v2#S2>
- **摘录要点：** High-Level Planning Layer（慢）：**Subtask Planner** 输出可执行子任务 $\ell_t^{\mathrm{sub}}$；**Tactile World Model** 预测视触觉子目标 $g_t$。中间层：**Visuo-Tactile Goal-Conditioned Policy** 生成名义动作块 $\hat{\mathbf{A}}_{t:t+H-1}$（$H=32$）。快层：**TRT** 在控制环内用近期触觉/本体历史对滑动名义窗口做残差修正（$W=16$，提交前 $C=4$ 步）。
- **对 wiki 的映射：**
  - [TouchWorld（论文实体）](../../wiki/entities/paper-touchworld-tactile-foundation-dexterous-manipulation.md) — 「预测 + 反应」双通路触觉基础模型读点。

### 2) 触觉世界模型：人视频接触先验 + 机端子目标预测

- **链接：** <https://arxiv.org/html/2607.07287v2#S3.SS2>
- **摘录要点：** 基于 **Wan2.2-TI2V-5B**；先在 **EgoTouch**（[TouchAnything](https://arxiv.org/abs/2605.13083) 人–物双手触觉估计数据）上预训练视觉→触觉动力学先验；再在 **10 h**（约 **1.08M** 帧 @30 FPS）机器人演示上微调，预测 **17 帧** $384\times224$ 视触觉子目标片段。Held-out 上 **Temporal Contact Acc. 86.3%**，显著优于 persistence / nearest-neighbor 基线。
- **对 wiki 的映射：**
  - [TouchWorld](../../wiki/entities/paper-touchworld-tactile-foundation-dexterous-manipulation.md) — 与 [T-Rex](../../wiki/entities/paper-trex-tactile-reactive-dexterous-manipulation.md)「未来视觉潜变量」、[视触觉融合](../../wiki/concepts/visuo-tactile-fusion.md)「接触后触觉主导」的 **显式触觉子目标预测** 轴。

### 3) 名义 VLA + TRT 残差：触觉图像进 VLA、结构化触觉历史进快层

- **链接：** <https://arxiv.org/html/2607.07287v2#S2.SS2>、<https://arxiv.org/html/2607.07287v2#S2.SS3>
- **摘录要点：** 名义策略把原始触觉 **渲染为图像** 与 RGB 同路进 VLA 骨干（兼容图像预训练）；TRT 则对 **图像式触觉图、矩阵触觉、低维触觉状态** 用轻量编码器 + 残差 Transformer，在 **58 维触觉敏感动作子空间** 上预测 $\Delta\mathbf{A}$，其余维度沿用名义 VLA 输出。
- **对 wiki 的映射：**
  - [视触觉融合](../../wiki/concepts/visuo-tactile-fusion.md)、[OmniTacTune](../../wiki/entities/paper-omnitactune-tactile-residual-adaptation.md) — 「残差精修」第三路线的人形灵巧 / 长程层级实例。

### 4) 四阶段分模块训练（非端到端反传全栈）

- **链接：** <https://arxiv.org/html/2607.07287v2#S3>
- **摘录要点：** Stage 1 Subtask Planner SFT（**128,866** 条语义更新率样本）；Stage 2 Tactile World Model（人预训练 + 机端微调）；Stage 3 名义 Visuo-Tactile Policy 模仿 + flow matching；Stage 4 冻结名义 VLA，训练 TRT 残差（目标为演示高频动作与名义窗口之差）。
- **对 wiki 的映射：**
  - [TouchWorld](../../wiki/entities/paper-touchworld-tactile-foundation-dexterous-manipulation.md) — 与 [VLA](../../wiki/methods/vla.md) 分阶段配方对照。

### 5) 六任务真机基准与人为扰动鲁棒性

- **链接：** <https://arxiv.org/html/2607.07287v2#S4>
- **摘录要点：** 六任务覆盖长程语义（浇花、清桌面）、精密插入（杯、插头）、持续接触（擦锅）与软物体（抽纸巾）。人为扰动含目标位移、不稳定接触、抓握干扰。消融：**去触觉** 降幅最大；去 TRT 特别伤害扰动设置；去 Subtask Planner 伤害长程一致性；去 Tactile World Model 削弱接触感知目标条件。
- **对 wiki 的映射：**
  - [接触丰富操作](../../wiki/concepts/contact-rich-manipulation.md)、[Manipulation](../../wiki/tasks/manipulation.md) — 2026 人形灵巧触觉层级策略 benchmark 族。

## 对 wiki 的映射（汇总）

- 沉淀实体页：[TouchWorld 预测–反应式触觉基础模型（arXiv:2607.07287）](../../wiki/entities/paper-touchworld-tactile-foundation-dexterous-manipulation.md)
- 交叉补强：[视触觉融合](../../wiki/concepts/visuo-tactile-fusion.md)、[接触丰富操作](../../wiki/concepts/contact-rich-manipulation.md)、[T-Rex](../../wiki/entities/paper-trex-tactile-reactive-dexterous-manipulation.md)、[OmniTacTune](../../wiki/entities/paper-omnitactune-tactile-residual-adaptation.md)、[VLA](../../wiki/methods/vla.md)
- 项目页归档：[`sources/sites/touchworld-phanes-lab.md`](../sites/touchworld-phanes-lab.md)

## 当前提炼状态

- [x] 四层架构、四阶段训练、TWM 人/机数据、TRT 调度、六任务结果与消融、项目页开源核查已摘录
- [x] 与 [`sources/sites/touchworld-phanes-lab.md`](../sites/touchworld-phanes-lab.md) 互证

## BibTeX

```bibtex
@misc{zhou2026touchworldpredictivereactivetactile,
      title={TouchWorld: A Predictive and Reactive Tactile Foundation Model for Dexterous Manipulation},
      author={Jianyi Zhou and Feiyang Hong and Yunhao Li and Yicheng Zhao and Yongjue Cen and Zirui Liu and Jiakang Huang and Zirui Chen and Ruiyang Zhang and Weizhuo Zhu and Xuhua Song and Shuo Yang},
      year={2026},
      eprint={2607.07287},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2607.07287},
}
```
