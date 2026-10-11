# AD-E2E-JEPA: A Joint-Embedding Predictive Architecture for End-to-End Autonomous Driving（arXiv:2609.34085）

> 来源归档（一手论文）

- **类型：** paper / autonomous driving / JEPA world model / zero-shot planning
- **arXiv：** <https://arxiv.org/abs/2609.34085>
- **PDF：** <https://arxiv.org/pdf/2609.34085>
- **作者：** Haoran Zhu、Wancong Zhang、Yann LeCun、Anna Choromanska
- **机构：** New York University；AMI Labs
- **提交日期：** 2026-09-28（本次核对的 arXiv 记录为 v1）
- **官方代码：** <https://github.com/HaoranZhuExplorer/AD-E2E-JEPA>（论文称有源码链接；仓库 README 当前称公开代码仍在准备中）
- **入库日期：** 2026-10-11
- **一句话说明：** 冻结 DINOv3 ViT-L，通过 stride-2 投影器和 patch 级 SIGReg 压缩世界模型表征，以动作条件潜变量 rollout 对目标图像进行轨迹搜索；256 候选约 0.8 秒/场景，8192 候选约 18.2 秒/场景换取更高 NAVSIMv2 分数。

## 核心论文摘录

### 1. 面向驾驶的 action-conditioned JEPA（§3.1–3.2）

冻结 DINOv3 ViT-L 编码历史及未来帧；相邻帧自车相对位姿作为动作条件，AdaLN 风格 predictor（RoPE）预测未来 patch embedding。基础预测损失在表征空间计算 MSE。

- **映射：** [AD-E2E-JEPA 实体页](../../wiki/entities/paper-ad-e2e-jepa.md)

### 2. 投影压缩与 SIGReg（§3.2.2、附录 A.1）

两层 stride-2 卷积将空间 patch 数减少 16 倍、特征维由 1024 降至 256；stop-gradient 投影目标与 MSE 配合 patch/time 位置级 SIGReg。附录配置使用 17 个 knots、1024 个随机方向。

- **映射：** [方法细节](../../wiki/entities/paper-ad-e2e-jepa.md#核心原理)

### 3. 目标条件零样本规划（§3.3–4.3、Table 2）

以未来帧图像作为目标，对驾驶轨迹词表候选进行世界模型 rollout，按预测终点与目标嵌入的平方距离选轨迹。12,146 场景全量测试中，trainval + rollout、256 候选得到 EPDMS/EPDMS† 67.3/84.1、FDE 4.0 m；8192 候选得到 72.9/86.5、FDE 2.8 m，时延由 0.8 秒升至 18.2 秒/场景。

- **映射：** [评测结果](../../wiki/entities/paper-ad-e2e-jepa.md#评测与结果)

### 4. 预训练 projector 的下游迁移（§3.4、§4.4、Table 3）

移除世界模型 predictor，保留 DINOv3 + 自监督预训练 projector，接轨迹 decoder 并微调；单视角、4 帧 NAVSIMv2 设置下 EPDMS 从随机 projector 的 80.2 提升至 85.4。

- **映射：** [迁移结果](../../wiki/entities/paper-ad-e2e-jepa.md#评测与结果)

## 结果解读边界

- zero-shot planning 的 goal 是 ground-truth future observation，测量的是给定目标时的轨迹选择，不是自主目标生成。
- EPDMS 含乘法安全项；EPDMS† 排除这些项，只反映加权驾驶质量分量。
- DINO-WM / JEPA-WM 的昂贵比较使用 100 场景子集；AD-E2E-JEPA 另有 12,146 场景全量结果，不能跨测试子集简单排名。
- 8192 候选的最高分对应 18.2 秒/场景，而非 0.8 秒设置。
- 论文 PDF 声明源代码可用；官方仓库 README 截至 2026-10-11 仍称代码准备公开。复现状态以[仓库核查](../repos/ad-e2e-jepa.md)为准。

## 关联知识节点

- [AD-E2E-JEPA 实体页](../../wiki/entities/paper-ad-e2e-jepa.md)
- [LeWM](../../wiki/entities/paper-lewm.md) — 像素端到端动作条件 JEPA
- [DINO-WM](../../wiki/entities/paper-sa-2411-04983-dino-wm-world-models-on-pre-trained-visual-featu.md) — 预训练视觉表征零样本规划

## 参考资料

- Zhu et al., “AD-E2E-JEPA: A Joint-Embedding Predictive Architecture for End-to-End Autonomous Driving,” arXiv:2609.34085, 2026. <https://arxiv.org/abs/2609.34085>
- 官方仓库：<https://github.com/HaoranZhuExplorer/AD-E2E-JEPA>
