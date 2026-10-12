# PLaW-VLA 论文来源归档

## 元数据

- **题目：** PLaW-VLA: Predictive Latent World Modeling for Vision-Language-Action Policies
- **作者：** Yu Liu、Hetian Guo、Tianlv Huang、Ziyi Cai、Wudi Chen、Hantang Wang、Qiutong Liu、Yingzhi Peng、Wei Han、Peijun Tang、Jianan Wang、Zipei Fan（通讯）、Zhiyuan Zha、Xuan Song
- **单位：** 吉林大学（Jilin University）；Astribot；哈尔滨工业大学（深圳）；香港理工大学；东京大学
- **预印本：** https://arxiv.org/abs/2610.12285
- **HTML：** https://arxiv.org/html/2610.12285v1
- **版本：** v1，2026-10-08，cs.RO；项目页 BibTeX 标注 CoRL 2026
- **项目页：** https://rainyrobo.github.io/PLaW-VLA/ （归档见 [sources/sites/plaw-vla.md](../sites/plaw-vla.md)）
- **代码：** https://github.com/RainyRobo/PLaW-VLA （归档见 [sources/repos/plaw_vla.md](../repos/plaw_vla.md)）

## 论文要解决的问题

VLA 主要依据当前观测输出动作，缺少对「动作之后环境会怎样」的显式预测；而像素级生成式世界模型预测未来视频又昂贵，且会把大量与动作无关的外观细节带进策略。PLaW-VLA 改在预训练的、面向预测的表征空间里预测任务相关的未来状态。

## 方法与训练信息

1. **未来表征：** 冻结 V-JEPA 2 编码器；目标是未来观测的 stop-gradient 编码。消融将其换成 Cosmos VAE 潜变量作对照。
2. **架构：** 三专家 Mixture-of-Transformers，按因果顺序——PaliGemma（3B VLM）视觉语言专家产生语义嵌入 φ_t；从零训练的潜在世界模型（LWM）专家预测未来潜变量；动作专家解码连续动作块。VLM 与动作专家由 π0.5 初始化，LWM 从零开始。相对 π0.5 约多 0.9B 参数、单张 H100 约 70 ms 推理延迟。
3. **并行未来预测：** K 个可学习未来查询 s_{1:K} 关注历史潜变量与 φ_t，一次前向预测全部 K 个未来潜变量（消融设置 K=2：近、远未来）；不生成也不解码视频。结构化因果注意力按「语义→潜变量预测→动作」顺序，避免未来信息泄漏。
4. **损失：** 潜变量预测为平方 L2（目标 stop-gradient）；动作为 flow matching（τ~Beta(1.5,1)，沿用 π0）；L_joint = L_FM + m·λ_wm·L_wm，m~Bernoulli(1−α)，α=0.3；m=0 时丢弃世界模型 token 与 L_wm，但保留动作损失。
5. **三阶段流程：** Stage I 在无动作的人与机器人视频上预训练 LWM；Stage II 在带动作的机器人轨迹上联合训练预测与动作；Stage III 针对目标任务/本体微调（学习率 5e-5，线性 warmup）。仓库给出的步数/批大小：Stage I 100k 步 / 512，Stage II 100k / 256，Stage III（LIBERO）50k / 256、warmup 10k。
6. **数据：** LIBERO 1,693 条示教（四套件）；RoboTwin 2.0 共 50 任务，2,500 条 clean（每任务 50）+ 25,000 条随机化（每任务 500）；真机每任务 400 条示教，双臂轮式人形。
7. 原文片段未给出动作块长度、历史长度、λ_wm、编码器分辨率、GPU 数（表 11 被截断），原文未核实。

## 论文报告的结果

- **RoboTwin 2.0（Easy / Hard）：** PLaW-VLA 平均 85.6 / 83.2；π0.5 82.7 / 76.8；X-VLA 72.9 / 72.8；π0 65.9 / 58.4。Horizon III（Easy / Hard）：PLaW-VLA 82.4 / 79.2，π0.5 78.6 / 67.4，Hard 上 +11.8 个百分点。逐任务 Hard：36 个提升、4 个持平、10 个下降。
- **LIBERO：** 平均 97.4（Spatial 98.9 / Object 99.0 / Goal 97.1 / Long 94.6），π0.5 96.9，VLA-JEPA 97.2，OpenVLA-OFT 97.1；VLA-JEPA 在 Goal 与 Long 上更高。
- **LIBERO-Plus 零样本：** 平均 72.7（Robot 73.5 / Layout 89.9 / Light 93.1 / Background 93.6 / Language 75.7 / Noise 55.2 / Camera 41.5）；OpenVLA-OFT 69.6、π0-FAST 61.6、π0 53.6。
- **消融（LIBERO / LIBERO-Plus）：** 完整 97.40/72.70；去 Stage I 96.28/69.05；去世界模型 96.05/65.49；去预测损失 96.30/67.56；去未来条件 96.85/69.81；Cosmos VAE 替代 V-JEPA 2 为 96.65/70.93；去 Stage II 95.65/66.03；去 I+II 94.05/60.80。推理期屏蔽未来状态：全屏蔽 93.05，屏蔽远期 95.45，屏蔽近期 95.80（LIBERO）。
- **延迟：** 相比 Motus 在相近 LIBERO 成功率下约 1/19 推理延迟。
- **真机（项目页，π0 / π0.5 / PLaW-VLA，%）：** Fold Towel 50/60/80；Play Basketball 47.5/42.5/50；Pack Toy 46.7/50/66.7；Organize Stationery 43.3/41.7/55；Fold Towel 零样本 30/50/80；Organize Stationery 零样本 35/40/48.3。论文正文仅以归一化任务进度图呈现。

## 复现与解释边界

- 论文自述局限：未来预测与闭环控制耦合，执行误差会反向影响预测；训练数据规模与多样性有限，视频→机器人范式的可扩展性未验证。
- 论文文本未做因果跨专家注意力的消融。
- 官方仓库已公开训练与评测代码，但官方预训练检查点截至 2026-10-12 标注「计划发布」，且论文确切数据混合依赖作者核验的来源清单。

## Robotics_Notebooks 映射

- 详情节点：wiki/entities/paper-plaw-vla-predictive-latent-world-model-vla.md
- 概念入口：wiki/methods/vla.md、wiki/concepts/world-action-models.md

## 一手来源

- arXiv 摘要：<https://arxiv.org/abs/2610.12285>
- arXiv HTML：<https://arxiv.org/html/2610.12285v1>
- 项目页：<https://rainyrobo.github.io/PLaW-VLA/>
- 代码：<https://github.com/RainyRobo/PLaW-VLA>
