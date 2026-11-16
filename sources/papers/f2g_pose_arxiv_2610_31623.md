# F2G-Pose — Geometry-Aware Foundation Feature Lifting for Direct RGB-D Category-Level Object Pose Estimation

> 原始论文深读归档（arXiv:2610.31623v1；NeurIPS 2026 accepted）

- **类型：** paper
- **标题：** F2G-Pose: Geometry-Aware Foundation Feature Lifting for Direct RGB-D Category-Level Object Pose Estimation
- **中文标题：** F2G-Pose：面向直接 RGB-D 类别级物体位姿估计的几何感知基础特征三维抬升
- **作者：** Jinyu Zhang、Haitao Lin、Jiashu Hou、Xiangyang Xue、Yanwei Fu
- **机构：** 复旦大学（Fudan University）；上海创智学院（Shanghai Innovation Institute）
- **提交：** 2026-10-22；论文标注 Accepted at NeurIPS 2026
- **原文：** https://arxiv.org/html/2610.31623v1
- **PDF：** https://arxiv.org/pdf/2610.31623
- **项目：** https://jrryzh.github.io/F2G-Pose/
- **代码：** https://github.com/jrryzh/F2G-Pose
- **权重：** https://huggingface.co/J3rr1/F2G-Pose
- **核对范围：** 已通读 arXiv v1 HTML 全文：摘要、§1–5（研究动机、相关工作、方法、各评测表格、机器人试验、局限）、附录 A.1–A.7 与 B.1–B.5（实现参数、训练、鲁棒性、GenPose++ 候选、失败模式、消融、匹配形状评测、mask 敏感性、专家路由、HANDAL 选择和 baseline 协议）。PDF 大文件在当前抓取接口未能打开，使用 arXiv 官方同版本 HTML 全文阅读，不宣称 PDF 逐页核验。
- **通读：** 已完成
- **入库日期：** 2026-11-09
- **对 wiki 的映射：** [F2G-Pose 项目实体](../../wiki/entities/paper-f2g-pose.md)

## 论文主张（§1–3）

- **设定：** 一次输入目标分割后的 RGB 图、对齐深度、相机内参；无需目标实例 CAD、参考图或显式类别标签，直接预测 6D 位姿（SO(3) 旋转与三维平移）、公制尺寸、相机坐标系下补全点云。它是按 SOPE 149 类训练的类别级方法，不等于对任意开放词汇类别零样本保证。
- **特征抬升：** 冻结 RADIOv2.5-L；特征层 8、16、23 用 1×1 卷积到 512 维，softmax 加权融合、3×3 卷积输出 1024 维；根据投影/patch=16 将图像特征分配给 3D 点，与坐标拼接（每点 1027 维）。
- **物体编码：** 采样 1024 个可见点 → DGCNN 两级降采样为 128 个 512 维 token → 8 层、8 头 Transformer，FFN 为 8 专家 top-2 MoE，负载均衡辅助损失 0.01 → 池化全局表示。
- **两支输出：** 旋转连续 6D 表示经正交化为旋转矩阵，同时回归平移与尺寸；形状分支先产生 512 个粗点，加入 256 个可见点后打分筛 512 个，逐点产生 4 个偏移，形成 2048 个相机系补全点。推理可跳过形状分支。
- **训练损失：** 旋转对称等价集合取最小 SmoothL1 + 平移 SmoothL1 + 尺寸 SmoothL1 + 粗/细 Chamfer-L1 + MoE load balancing。训练只使用带姿态标注的合成 SOPE；RADIO 有外部预训练，不能误称整个系统从零仅使用 SOPE 训练。

## 三组核心原文摘录与证据映射

### 1. 直接预测 vs 候选采样

- 论文 §1–3、Table 2 与 Fig.3：GenPose++ 扩散得到候选后仍需 scoring/filter/cluster；在 ROPE best-of-50 oracle 近真候选比例 79.3%（≤5°），最终流水线 48.3%，因此候选质量≠部署决策质量。
- **对 wiki 的映射：** [F2G-Pose 核心原理与方法对比](../../wiki/entities/paper-f2g-pose.md)。

### 2. 主结果及消融

- **论文 Table 1（历史论文协议）：** ROPE IoU50 AUC F2G-Pose **27.8** vs GenPose++ **19.1**；SOPE **43.7** vs **31.9**。ROPE 平均旋转误差 27.5°，平移 1.19 cm。
- **论文 Table 4：** 完整 27.8（23.7 FPS），w/o RGB 16.2（29.5 FPS），w/o MoE 24.0（19.2 FPS），w/o shape supervision 23.2（27.8 FPS）；这是消融对照，不是每项对实际真机复现的证明。
- **论文 Table 3：** SOPE/HouseCat6D camera-frame Chamfer-L1（×10^-3）分别 5.42/7.1。附录 B.1 提醒不同体系输入与监督不同，需查看匹配 RGB-D canonical-frame 对照，不能直接将不同坐标尺度 CD 当公平值。
- **对 wiki 的映射：** [F2G-Pose 评测与局限](../../wiki/entities/paper-f2g-pose.md)。

### 3. 实时与 ARX 机器人试验

- **论文 Table 9：** ARX 机械臂、平行夹爪、外置 RealSense、相同 RGB-D+mask 输入；每方法 80 次随机摆放试验，完整模型平均感知延迟 **36 ms**、50 ms 内 ready **100%**、抓取 **74/80**、放置 **68/80**；补全形状仅用于 collision checking，不能说网络直接输出 grasp trajectory。
- **论文 Appendix A.2：** 四张 H200、PyTorch 2.4/CUDA 12.4、100 epochs、batch 512；RTX 4090 batch 1 GPU 同步测完整模型 **23.7 FPS**、pose-only **28 FPS**（分割与部分前处理不计）。
- **对 wiki 的映射：** [F2G-Pose 工程部署](../../wiki/entities/paper-f2g-pose.md)。

## 补充评测 / 数据边界

- ROPE 是相同类目、未见实例的真实扫描物体评测；HANDAL 按分类映射筛选出 7 类 92 实例，报告 IoU50 11.7；不能泛化为所有开放类别。
- 透明/反光物体仍有深度缺失；mask/深度噪声敏感。附录 B.2：ROPE GT mask IoU50 27.8、SAM mask 27.2、80% IoU erosion 23.8；深遮挡与对称旋转仍是失败源。
- **重要版本区别：** 官方 GitHub/Hugging Face 现提供同一检查点的 full-test 快照，SOPE AUC50 44.73、ROPE AUC50 28.19（docs/EVALUATION.md）；它与论文表 1 的 43.7/27.8 采用不同覆盖/评测快照，不应混为单个实验。
- **当前源码 README 测速：** RTX 4090 batch 1 FP32 模型计时 36.6 FPS（完整）、37.2 FPS（pose-only），30 次 warmup + 200 次同步迭代，不含预处理/SAM；与论文 23.7/28 采用不同版本/条件，不互相替代。

## 对 wiki 的映射

- 主节点：[F2G-Pose](../../wiki/entities/paper-f2g-pose.md)
- 项目主页归档：[F2G-Pose 项目站](../sites/f2g-pose.md)
- 源码复核归档：[jrryzh/F2G-Pose](../repos/jrryzh_f2g_pose.md)

## 参考资料

- [arXiv HTML v1](https://arxiv.org/html/2610.31623v1) — 2026-10-22 全文
- [官方仓库 README](https://github.com/jrryzh/F2G-Pose)
- [官方评测协议](https://github.com/jrryzh/F2G-Pose/blob/main/docs/EVALUATION.md)
