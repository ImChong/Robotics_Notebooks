# BATCH: Behavior Alignment for Cross-Embodiment Learning from Humans（arXiv:2611.10148）

> 一手论文归档；以 arXiv PDF v1 全文为依据，非二手转述。

- **标题：** BATCH: Behavior Alignment for Cross-Embodiment Learning from Humans
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2611.10148>
- **PDF：** <https://arxiv.org/pdf/2611.10148v1>
- **项目页：** <https://batch-embodiment.github.io/>
- **作者与机构：** Zeyi Liu、Shuran Song；Stanford University（论文首页）
- **版本：** arXiv:2611.10148v1 [cs.RO]，2026-11-06
- **代码：** 未开源。项目页 Code 按钮标 “coming soon”；论文可复现性声明称“接受后”释出代码、数据集与检查点（2026-11-10 核查）。
- **入库日期：** 2026-11-10
- **一句话说明：** 用 GTCC（多假设时间循环一致性）加子序列 DTW 在无标注下对齐人类与机器人示范的行为阶段，再在自回归视频-动作扩散模型里做对齐联合训练与人到机器人风格迁移；两项真机任务的 OOD 表现优于不对齐与既有基线。
- **核对范围：** 通读 arXiv PDF v1 全部正文 §1–§5、致谢、AI 使用 / 伦理 / 可复现性声明、参考文献列表，以及附录 A（数据采集）、B（方法细节与 Table 2）、C（对齐评测协议）、D（逐场景结果）；Table 1 与 Figure 4 / 7 的数值经文本提取读取，Figure 4 柱状图的柱子与方法对应关系依据坐标位置与正文描述交叉推断。未运行任何代码，未阅读所引的 GTCC（Donahue & Elhamifar, 2024）、DreamZero、EgoVerse 原文；项目页只读了页头与 Code 状态，视频未看。
- **通读：** 已完成

## 方法要点

1. **行为对齐（§3.1）：** 冻结 ImageNet ResNet-50 提取每帧 1024×14×14 特征，因果时间窗 k 帧堆叠，两层 3D 卷积、时空最大池化、三层 MLP 得到 128 维嵌入。用 GTCC 的多假设软最近邻（K 分量高斯混合）与多循环一致性损失 `ℓ_i = Σ_k π_ik (i-μ̂_ik)² + λ log σ̂_ik`，辅助头预测相关度 `b_i`，总损失 `L_i = b_i ℓ_i + (1-b_i) ℓ_i^{-1}`，让无对应帧被降权。
2. **检索（§3.1.2、附录 B.1）：** 在学到的嵌入上做子序列 DTW，选终点 `j* = argmin D[L, j]`，按归一化 DTW 代价取 top-k（Object in Container 取 k=5，Cup on Saucer 取 k=1）；训练时每步均匀采样一个候选，沿 warping path 取人类帧、状态与动作。
3. **视频-动作模型（§3.2）：** 基于 DreamZero 风格的视频-动作扩散 Transformer，骨干为 Wan2.1-I2V-14B 的前 8 个 Transformer 块 + 视频 VAE + CLIP ViT-H/14；去掉文本条件（单任务）；9 帧头部相机、16 步动作块；人类流与机器人流共享权重、注意力隔离、RoPE 与 CLIP 交叉注意力按本体分开。
4. **训练目标（§3.2.1、附录 B.2.3）：** 机器人预训练 20,000 步（4 张 RTX PRO 6000 Blackwell 96GB，批大小每卡 16，学习率 1e-4）；微调 1,000 步（学习率 2e-5，层衰减 0.95），每步以 0.5 概率在联合协同训练 `L_joint` 与风格迁移 `L_ST` 间选择。`L_ST` 以完整人类片段为干净上下文，只监督机器人未来视频，并遮蔽机器人条件潜变量与动作 / 状态寄存器。
5. **部署（§3.2.1、附录 B.2.4）：** 只用机器人流；16 步动作块执行前 8 步（10 Hz）再重规划；16 步 UniPC，8 次网络评估；RTX 5090 上延迟 0.35 s（单臂）/ 0.42 s（双臂）。

## 实验设置与结果

- **平台与数据（附录 A、B Table 2）：** 人类用 Aria Gen 1 眼镜（手位姿来自 MPS，7 维 / 手，无夹爪宽度，补零并从人类动作损失中排除）；机器人为两个 ARX5 臂，头部装 Aria Gen 1，腕部各一个 Logitech 摄像头，动作 8 维 / 臂。Object in Container（单臂）：机器人预训练 90 条演示，人类微调每场景 5 条；Cup on Saucer（双臂）：270 条机器人演示，人类每场景 8 条。
- **对齐评测（§4.1、Table 1）：** 阶段重叠率与阶段进度误差。Object in Container：GTCC+DTW 重叠 ID 69.0% / OOD 57.8%，进度误差 .061 / .098；去掉 DTW 后 OOD 重叠降到 24.7%，误差 .231；DINOv3+DTW OOD 重叠 42.3%。Cup on Saucer：GTCC+DTW OOD 重叠 71.7%（DINOv3+DTW 22.1%）；GTCC+绝对位姿最好（ID 79.2%、OOD 76.5%）。结论：位姿线索仅在人机轨迹空间对应较好时有益（作者原话）。
- **策略评测（§4.2、Figure 4）：** Object in Container 为 4 个 OOD 配置共 20 次 60 秒滚动，分数按域内平均（约 55 分）归一化：OURS 0.63、OURS w/ OT 0.50、OURS w/o alignment 0.42、EgoVerse 0.40、Robot only 0.28。Cup on Saucer 为 30 次 OOD 滚动：OURS 83.3%、w/ OT 63.3%、w/o alignment 50.0%、EgoVerse 63.3%、Robot only 33.3%（柱子与方法的对应由位置与正文推断，正文明确 w/ OT 与 EgoVerse 在该任务持平）。
- **作者汇总（§1）：** 对 co-training 平均提升 45%，对仅机器人预训练模型平均提升 138%。核对：由 Figure 4 的数值算出 Object in Container 相对 EgoVerse +57.5%、Cup on Saucer +31.6%，均值约 44.5%；相对 Robot only 为 +125% 与 +150%，均值约 137.5%，与作者口径一致。
- **逐场景（附录 D、Figure 7）：** 每个 Object in Container 场景 5 次，每个 Cup on Saucer 场景 10 次滚动；OURS 在所有场景最强。

## 作者承认的局限（§4.2 末段「Limitation」）

- 对齐方法假设人机示范共享行为结构，策略或动作顺序差异大时可能失败；设想用 VLM 粗粒度语义对齐加帧级精修。
- 对齐误差会把错配监督带入风格迁移目标。
- 后训练阶段仍需要机器人示范，降低该依赖留作未来工作。

## 对 wiki 的映射

- [paper-batch-behavior-alignment](../../wiki/entities/paper-batch-behavior-alignment.md)
