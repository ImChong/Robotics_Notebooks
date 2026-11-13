# DexTacWAM: A Visuo-Tactile World-Action Model for Dexterous Manipulation — 原文通读

- **中文标题：** DexTacWAM：面向灵巧操作的视触觉世界—动作模型
- **类型：** 一手论文
- **作者：** Haoran Yuan、Zekai Wang、Boning Shao、Haoran Lu、Trevor Darrell、Ismini Lourentzou、Wei Zhan
- **机构：** 伊利诺伊大学厄巴纳-香槟分校（UIUC）、加州大学伯克利分校（UC Berkeley）、美国西北大学（Northwestern）
- **原文：** https://arxiv.org/abs/2609.24976v1
- **PDF：** https://arxiv.org/pdf/2609.24976
- **HTML：** https://arxiv.org/html/2609.24976v1
- **版本：** arXiv:2609.24976v1（2026-09-21；22 页）
- **项目页：** https://dextacwam.github.io/
- **代码：** https://github.com/dextacwam/DexTacWAM
- **核对范围：** 论文 `1–2 引言与相关工作、`3.1–3.5 逐指编码/压缩/联合世界建模/动作模型/分阶段适配（公式 1–5、Figure 2–3）、`4.1–4.3 六任务实验/消融/效率/预测保留（Tables 1–5）、`5–6 结论及局限；附录 A.1–A.11 的评分协议、基线、统计检验、编码器细节、训练配方、消融、失败分析与可视化，参考文献和插图。
- **通读：** 已完成

## 原文核对摘录

1. **编码与压缩（`3.1–3.2，A.5）：** 十个灰度指尖触觉图由 1×1 GrayToRGB 接入冻结的视觉 LTX VAE，逐指编码后加入手指 ID、手姿态、hand query；self-attention pooling 将每手五指缩为一个 latent，TimeSformer 处理时序，融合前后形变重建约束接触信息保留。
2. **联合世界状态（`3.3）：** 视觉 latent 与左右手触觉 latent 沿 view 轴合并，用分模态 flow matching loss 训练视频 DiT 预测视觉与接触演化。动作专家使用单次 world-model 前向的预测特征，而非完整未来图像生成。
3. **动作模型（`3.4，A.7）：** 视觉/触觉 K/V 分模态无参数 RMS 归一化；统一动作目标包括十指各 6D 力相关目标（共 60D）、相对末端位姿、手目标和本体状态，总目标 194D。**力在训练目标中，而不是部署期额外真实力输入。**
4. **三阶段（`3.5，A.6）：** Stage 1 用 488 条数据约 4h（仓库 README 口径为 3.5h）训练可迁移触觉编码器；Stage 2 不做视频模型的触觉 midtraining，直接以各任务约 100 条示范微调联合世界模型；Stage 3 用同任务数据从头训练动作专家。
5. **真机评价（Table 1，A.1）：** 22-DoF 双臂平台六任务分别得 90.0/71.6/77.0/60.0/65.0/60.0，平均 70.6，对照最强 RDP 平均 38.0。每项每模型 20 次实验；Handover 与 Wipe 采用分阶段评分，其余四任务为二元评分，故平均是**任务分数而非统一二元成功率**。
6. **消融（Tables 2–3）：** 同触觉特征和数据、去掉触觉 world modeling，四任务平均 74.7→26.6；去掉模态 K/V RMS 后动作专家不收敛。压缩后接触 recall 保留率 89.4%，不是操作成功率。
7. **效率与质量（`4.2–4.3）：** RTX 4090 上 3168→864 tokens，迭代 3.87→1.71 s（2.26×），动作块推理 363.0→281.6 ms（1.29×）；触觉预测 cosine 0.967、contact F1 0.737。与视觉-only 对照的 PSNR 差值在 Handover / Wipe 是 +0.06/−0.07 dB，仅对应匹配的两任务。
8. **局限（`6，A.9）：** 只验证 Sharpa 视觉式触觉及对应本体；主要来自成功示范，失败后恢复与其它触觉原理未获验证。

## 对 wiki 的映射

- [DexTacWAM 主实体](../../wiki/entities/paper-dextacwam.md)

> 以上为论文一手报告，未在本环境复现真机。