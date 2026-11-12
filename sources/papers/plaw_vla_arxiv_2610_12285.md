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
- **核对范围：** 通读 arXiv:2610.12285v1 PDF 全文（首页标题与编号已核对）：摘要、§1 引言、§2 相关工作、§3 方法（式 1 到式 5）、§4 实验（表 1 到表 4、图 4 与图 5）、§5 结论、§6 局限，以及附录 A（表 5 到表 7）、B（表 8 到表 10、图 6）、C（真机任务与评分）、D（表 11、表 12、式 6）；图 2 和图 3 仅看文字说明与布局，图 4 的柱状图数值取自项目页（已打开核对）；未读参考文献逐条内容，未运行任何代码，未复核仓库源码。
- **通读：** 已完成

## 论文要解决的问题

VLA 主要依据当前观测输出动作，缺少对「动作之后环境会怎样」的显式预测；而像素级生成式世界模型预测未来视频又昂贵，且会把大量与动作无关的外观细节带进策略。PLaW-VLA 改在预训练的、面向预测的表征空间里预测任务相关的未来状态。

## 方法与训练信息

1. **未来表征：** 冻结 V-JEPA 2 编码器；目标是未来观测的 stop-gradient 编码。消融将其换成 Cosmos VAE 潜变量作对照。
2. **架构：** 三专家 Mixture-of-Transformers，按因果顺序——PaliGemma（3B VLM）视觉语言专家产生语义嵌入 φ_t；从零训练的潜在世界模型（LWM）专家预测未来潜变量；动作专家解码连续动作块。VLM 与动作专家由 π0.5 初始化，LWM 从零开始。相对 π0.5 约多 0.9B 参数、单张 H100 约 70 ms 推理延迟。
3. **并行未来预测：** K 个可学习未来查询 s_{1:K} 关注历史潜变量与 φ_t，一次前向预测全部 K 个未来潜变量（消融设置 K=2：近、远未来）；不生成也不解码视频。结构化因果注意力按「语义→潜变量预测→动作」顺序，避免未来信息泄漏。
4. **损失：** 潜变量预测为平方 L2（目标 stop-gradient）；动作为 flow matching（τ~Beta(1.5,1)，沿用 π0）；L_joint = L_FM + m·λ_wm·L_wm，m~Bernoulli(1−α)，α=0.3；m=0 时丢弃世界模型 token 与 L_wm，但保留动作损失。
5. **三阶段流程：** Stage I 在无动作的人与机器人视频上预训练 LWM；Stage II 在带动作的机器人轨迹上联合训练预测与动作；Stage III 针对目标任务/本体微调（学习率 5e-5，线性 warmup）。仓库给出的步数/批大小：Stage I 100k 步 / 512，Stage II 100k / 256，Stage III（LIBERO）50k / 256、warmup 10k。
6. **数据：** LIBERO 1,693 条示教（四套件）；RoboTwin 2.0 共 50 任务，2,500 条 clean（每任务 50）+ 25,000 条随机化（每任务 500）；真机每任务 400 条示教，双臂轮式人形。
7. 全文（含表 11、附录 B.4）给出：6 帧片段切成 3 个 2 帧时间单元（当前加两个未来目标），K=2（近、远未来）；所有数据重采样到 10 Hz；动作统一为每臂 8 维末端位姿加夹爪（双臂 16 维）；LIBERO 结果取 30k 步检查点（训练 50k 步）；真机 Stage III 约 50 epoch、warmup 1 epoch。全文未给出动作块长度 Ha 的数值、λ_wm 的数值、编码器输入分辨率、GPU 数量（正文仅称用混合精度与梯度累积凑全局 batch）。

## 论文报告的结果

- **RoboTwin 2.0（Easy / Hard）：** PLaW-VLA 平均 85.6 / 83.2；π0.5 82.7 / 76.8；X-VLA 72.9 / 72.8；π0 65.9 / 58.4。Horizon III（Easy / Hard）：PLaW-VLA 82.4 / 79.2，π0.5 78.6 / 67.4，Hard 上 +11.8 个百分点。逐任务 Hard：36 个提升、4 个持平、10 个下降。
- **LIBERO：** 平均 97.4（Spatial 98.9 / Object 99.0 / Goal 97.1 / Long 94.6），π0.5 96.9，VLA-JEPA 97.2，OpenVLA-OFT 97.1；VLA-JEPA 在 Goal 与 Long 上更高。
- **LIBERO-Plus 零样本：** 平均 72.7（Robot 73.5 / Layout 89.9 / Light 93.1 / Background 93.6 / Language 75.7 / Noise 55.2 / Camera 41.5）；OpenVLA-OFT 69.6、π0-FAST 61.6、π0 53.6。
- **消融（LIBERO / LIBERO-Plus）：** 完整 97.40/72.70；去 Stage I 96.28/69.05；去世界模型 96.05/65.49；去预测损失 96.30/67.56；去未来条件 96.85/69.81；Cosmos VAE 替代 V-JEPA 2 为 96.65/70.93；去 Stage II 95.65/66.03；去 I+II 94.05/60.80。推理期屏蔽未来状态：全屏蔽 93.05，屏蔽远期 95.45，屏蔽近期 95.80（LIBERO）。
- **逐任务 RoboTwin（附录 A.3，表 7）：** 相对 π0.5，Hard 上 36 个任务更高、4 个持平、10 个更低，Horizon III 的 5 个 Hard 任务全部提升；Handover Mic（Hard 97 到 81）和 Place Dual Shoes（Hard 75 到 50）退步。
- **预训练效果随检查点变化（附录 B.1，表 8、表 9）：** LIBERO 上完整模型相对「无 Stage I-II」的优势在 30k 步为 +3.35，50k 步缩到 +1.00，且 LIBERO-Long 在 50k 步反超（91.7 对 93.2）；LIBERO-Plus 50k 步优势 +2.83，但 Noise 一项为 -8.10。Stage III 在约 30k 步后进入平台（附录 B.2）。
- **推理期屏蔽未来状态逐套件（附录 B.4，表 10）：** 屏蔽全部未来状态使 LIBERO-Spatial 从 98.90 降到 89.60，而 LIBERO-Goal 为 97.40 对完整模型 97.10，即并非每个套件都受益；PCA 可视化仅为定性展示，不度量预测精度。
- **α 敏感性（附录 B.3）：** α 从 0.3 降到 0.1，30k 步 LIBERO 同为 97.4%，训练时间约增加 10%。
- **延迟（§4.4，图 5）：** 相比 Motus 在相近 LIBERO 成功率下约 1/19 推理延迟。
- **真机（§4.2、附录 C，图 4 为柱状图；下列数值取自项目页，π0 / π0.5 / PLaW-VLA，%）：** Fold Towel 50/60/80；Play Basketball 47.5/42.5/50；Pack Toy 46.7/50/66.7；Organize Stationery 43.3/41.7/55；Fold Towel 零样本 30/50/80；Organize Stationery 零样本 35/40/48.3。论文正文以「归一化任务进度」报告（Fold Towel 为二值，等同成功率；其余按得分点折算），每个设置 10 个空间配置乘 3 次共 30 次试验，π0、π0.5 与本文使用相同的每任务 400 条示教和训练预算（附录 C.2）；项目页标题写作 success rate，与论文的 task progress 口径不完全一致。论文未给出置信区间。

## 复现与解释边界

- 论文自述局限：未来预测与闭环控制耦合，执行误差会反向影响预测；训练数据规模与多样性有限，视频→机器人范式的可扩展性未验证。
- 论文文本未做因果跨专家注意力的消融（图 3 的 w/o Attention 与 w/ Attention 仅为示意，Table 4 无对应行）。
- 消融 (a)「w/o world model」同时去掉 Stage I（与 w/o Stage I 变体对比），故相对完整模型的 -7.21 包含去掉视频预训练的部分；与匹配基线相比仅为 -3.56（LIBERO-Plus）与 -0.23（LIBERO）。
- LIBERO-Plus 对比只含官方榜单中被列入的零样本基线（表 2、表 6）；Language、Noise、Camera 三项并非最优（如 Camera 41.5 低于 π0-Fast 的 65.1，Noise 55.2 低于 π0 的 79.0）。
- 1/19 延迟的比较（图 5）基于单张 H100 与 LIBERO 成功率「相近」；正文未列出 Motus 与本文延迟的具体毫秒数（图中对数轴）。
- 官方仓库已公开训练与评测代码，但官方预训练检查点截至 2026-10-12 标注「计划发布」，且论文确切数据混合依赖作者核验的来源清单。

## Robotics_Notebooks 映射

- 详情节点：wiki/entities/paper-plaw-vla-predictive-latent-world-model-vla.md
- 概念入口：wiki/methods/vla.md、wiki/concepts/world-action-models.md

## 一手来源

- arXiv 摘要：<https://arxiv.org/abs/2610.12285>
- arXiv HTML：<https://arxiv.org/html/2610.12285v1>
- 项目页：<https://rainyrobo.github.io/PLaW-VLA/>
- 代码：<https://github.com/RainyRobo/PLaW-VLA>
