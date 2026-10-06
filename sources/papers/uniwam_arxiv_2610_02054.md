# UniWAM: Unified World-Action Model

> 来源归档（paper；核对 arXiv HTML v1、项目页与官方代码仓库；2026-10-06）

- **arXiv：** <https://arxiv.org/abs/2610.02054> · [HTML v1](https://arxiv.org/html/2610.02054v1) · [PDF](https://arxiv.org/pdf/2610.02054)
- **项目页：** <https://uniwam.github.io/>
- **官方代码：** <https://github.com/UniWAM/UniWAM>（Apache-2.0）
- **检查点：** <https://www.modelscope.cn/collections/Kosmos524/UniWAM>
- **作者：** Jiayi Chen, Wenxuan Song, Jingbo Wang, Shuai Zhou, Xicheng Gong, Zehua Fan, Ziyang Zhou, Junwu E, Haodong Yan, Fuhao Li, Qize Yu, Xu Huang, Pengwei Wang, Wen Chen, Shunbo Zhou, Haoang Li
- **机构：** The Hong Kong University of Science and Technology (Guangzhou), OLA Dimensions, Carnegie Mellon University, Peking University, Shanghai Jiao Tong University, Beijing Academy of Artificial Intelligence
- **项目角色（官网）：** Jiayi Chen、Jingbo Wang、Shuai Zhou、Xicheng Gong 为 core contributors；Wenxuan Song 为 project lead；Shunbo Zhou 为 project PI。
- **一句话说明：** 以 MoT 联合物理语言推理、未来视觉生成和动作预测，在 VQA、人类第一视角视频和机器人示教上按专家分配互补监督；后训练引入未来视觉噪声和历史条件 flow matching。

## 核心摘录

1. **模型结构。** 官方代码将 UniWAM 定义为约 8B 参数 MoT：物理推理专家使用 Qwen3-VL-2B-Instruct；世界生成专家基于 Wan2.2-TI2V-5B，在冻结的视频自编码器 latent 中预测未来视觉；动作专家对连续动作块做 flow matching。三专家通过 joint multimodal attention 交换信息。
2. **三类数据的分工。** VQA 监督物理推理专家；人类 egocentric 数据监督推理与视频世界专家；机器人数据则为三个专家提供监督。数据清洗包含轨迹异常/状态动作时序、几何坐标一致性与无效视觉观测检查。
3. **人类视频标注。** EgoANT 先用全片 contact sheets 做粗分段，再在局部时间窗细化操作边界，并为每段生成动作描述。最终将语言、腕部轨迹和相机参考坐标时间对齐。
4. **后训练变化。** 以 0.5 概率额外扰动 future visual latent、保持当前帧干净；动作 flow 的 source 从纯高斯噪声改成“上一动作块 + 少量高斯噪声”，让近期动作历史初始化下一动作生成。
5. **训练语料。** 表 1 列出约 4,958 小时 robot + 5,072 小时 human（总计约 10,013 小时），另有 8.293M VQA QA pairs。正文 §3.1.1 又将 robot data 写作约 4,363 小时，与表 1 的小计不一致；这里保留这一原文口径差异。
6. **仿真结果。** 报告 LIBERO 平均 SR 99.2%；LIBERO-Plus overall 92.6%；RoboTwin 2.0 Clean2Clean 75.14%、Clean2Rand 68.32%。Clean2Rand 是域随机化条件，不能与 clean 设置混称。
7. **真机结果。** 双 AgileX Piper、腕部相机与第三视角相机；四项真实指令任务平均 SR 67.5%、IFR 82.5%。桌面整理长程任务平均进度 5.0/6；这些是论文自有实验协议，不是公开推理代码已经覆盖的场景。
8. **公开代码范围。** GitHub 提供 RoboTwin 与 LIBERO 代码、模型和 checkpoint 指引；README 明确 Bridge/DROID/Fractal 和 real-world inference 不属于当前 release。RoboTwin 后训练约 8×H100、10 小时；这不是完整预训练成本估计。

## Wiki 映射

- [UniWAM 独立论文/项目详情节点](../../wiki/entities/paper-uniwam-unified-world-action-model.md)
- [World Action Models 概念页](../../wiki/concepts/world-action-models.md)
- [官方代码来源归档](../repos/uniwam.md)

## 原始来源

- arXiv HTML v1：<https://arxiv.org/html/2610.02054v1>
- 官方项目页：<https://uniwam.github.io/>
- 官方 GitHub：<https://github.com/UniWAM/UniWAM>
- ModelScope：<https://www.modelscope.cn/collections/Kosmos524/UniWAM>
