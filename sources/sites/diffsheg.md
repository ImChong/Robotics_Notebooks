# DiffSHEG 项目页

> 来源归档（site / project page）

- **标题：** DiffSHEG: A Diffusion-Based Approach for Real-Time Speech-driven Holistic 3D Expression and Gesture Generation
- **类型：** project page
- **URL：** <https://jeremycjm.github.io/proj/DiffSHEG/>
- **论文：** <https://arxiv.org/abs/2401.04747>
- **代码：** <https://github.com/JeremyCJM/DiffSHEG>
- **视频：** <https://www.youtube.com/watch?v=HFaSd5do-zI>
- **机构：** 香港科技大学（HKUST）+ 国际数字经济学院（IDEA）
- **会议：** CVPR 2024
- **核查日期：** 2026-07-31
- **一句话说明：** DiffSHEG 项目页展示语音驱动的整体 3D 表情+手势联合扩散生成：UniEG-Transformer（表情→手势单向信息流）与 FOPPAS 任意长实时采样，并给出 BEAT / SHOW 定量对比与用户偏好结果。

## 开源状态（项目页核查，2026-07-31）

- 项目页提供 Abstract、Framework、Arbitrary-long Sampling（FOPPAS）、User Study、Quantitative Comparison 与 BibTeX。
- **代码入口**指向官方仓 [JeremyCJM/DiffSHEG](https://github.com/JeremyCJM/DiffSHEG)（BSD-3-Clause）；checkpoint 在仓库 README 的 Google Drive 链接。
- **结论：已开源**（训练/推理代码 + 权重下载；原始 BEAT/SHOW 数据需另取）。详见 [`sources/repos/diffsheg.md`](../repos/diffsheg.md)。

## 核心摘录（归纳，非全文）

- **问题：** 既有工作多单独做共语手势或表情；联合生成常拆成独立模型或多任务头，忽视表情–手势联合分布。
- **方法：** 统一扩散去噪网络 + **UniEG**（表情→手势单向条件流，梯度 detach）；测试时 **FOPPAS**（Repaint 式 outpainting + DDIM25）做任意长/流式采样。
- **实时性主张：** 单卡 RTX 3090 约 **31.5 FPS**（含 Mel + HuBERT；1 分钟 BEAT 音频约 28.6 s）。
- **数据：** BEAT（15 fps，34 帧训练窗）与 SHOW / TalkSHOW（SMPLX，30 fps，88 帧窗）。
- **对照：** BEAT 上 vs CaMN / DiffGesture / DiffuseStyleGesture / LDA；SHOW 上 vs LS3DCG / TalkSHOW；用户研究在 realism / sync / diversity 上主导偏好。

## 对 wiki 的映射

- [DiffSHEG 实体页](../../wiki/entities/paper-diffsheg.md)
- [扩散运动生成](../../wiki/methods/diffusion-motion-generation.md)
- [扩散模型](../../wiki/concepts/diffusion-model.md)

## 参考来源（原始）

- 项目页：<https://jeremycjm.github.io/proj/DiffSHEG/>
- arXiv：<https://arxiv.org/abs/2401.04747>
- 代码：<https://github.com/JeremyCJM/DiffSHEG>
