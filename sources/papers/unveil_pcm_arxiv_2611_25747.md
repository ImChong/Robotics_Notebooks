# UNVEIL：SAM 3 Sees More Than It Detects: Promptable Concept Maps Beyond Categories（arXiv:2611.25747）

> 来源归档（ingest · 论文全文核对）

- **标题：** SAM 3 Sees More Than It Detects: Promptable Concept Maps Beyond Categories
- **中文标题：** SAM 3 看见的不止于检测：超越对象类别的可提示概念图
- **类型：** paper / promptable-concept-maps / semantic-segmentation
- **论文：** <https://arxiv.org/abs/2611.25747>；[HTML 全文 v1](https://arxiv.org/html/2611.25747v1)；[PDF](https://arxiv.org/pdf/2611.25747)
- **作者：** Yu-Huan Wu、Yun Liu、Liangli Zhen
- **机构：** 新加坡科技研究局前沿人工智能研究中心（CFAR, A*STAR）；南开大学（Nankai University）
- **发布日期：** 2026-11-19（arXiv v1）
- **代码：** <https://github.com/yuhuan-wu/UNVEIL>；[仓库核对归档](../repos/unveil.md)
- **模型权重：** <https://github.com/yuhuan-wu/UNVEIL/releases/tag/v1.0>
- **入库日期：** 2026-11-21
- **核对范围：** arXiv 2611.25747v1 HTML 论文全文：摘要、引言、相关工作、问题形式化、方法（语义路径/存在判断/联合查询）、训练数据与损失、主要实验/对照/消融、限制及附录；并结合作者仓库 README、推理入口和核心源码交叉核实。
- **通读：** 已完成
- **一句话说明：** 利用 SAM 3 已有但被实例检测中的 presence gating 遮挡的语义路径，通过少量可训练拷贝与独立存在判断器，将文本短语映射为阴影、镜面、透明物体、显著/伪装目标、边缘/骨架等概念图或空结果。

## 原文摘录 1：任务定义与 SAM 3 隐藏的语义能力

- **Promptable Concept Maps（PCM）：** 同一接口接收自然语言概念提示，返回该概念在图像中的软像素图；概念不存在时应答「none」。
- **四类概念：** Identity（对象类别）、Light（shadow、transparent object、mirror）、Context（salient/camouflaged object）、Structure（edges、skeleton、symmetry axis）。
- **问题根源：** SAM 3 原有实例路径的检测得分同时受实例与图像级 presence 影响；对阴影/边缘等非对象概念，presence 可拒绝具备信息量的语义输出。论文给出 COD10K 的 camouflage 和 BSDS500 的 edges 作为对照。
- **对 wiki 的映射：** [UNVEIL 独立论文实体](../../wiki/entities/paper-unveil-pcm-2611-25747.md)；与 [SAM 3](../../wiki/entities/paper-sam3.md) 区分“可提示实例”与“可提示像素概念”。

## 原文摘录 2：复用视觉骨干，训练语义与存在分支

- 保留 SAM 3 共享视觉骨干前段与文本编码器；复制 ViT 最后 8 个 block，LoRA 微调；基于拷贝的语义/融合头输出 **288×288** coarse 概念图。
- 单独训练 presence decoder 处理概念「是否存在」，输入中的 memory/text 在该分支使用 stop-gradient，避免存在判别梯度伤害像素映射。
- 推理时将 coarse logits 插值到原图，使用 sigmoid 生成概念强度；presence 低于任务阈值时输出全零图。显著目标与伪装目标可做联合查询。
- 保留 SAM 3 原始分支与权重，因此同一实例中的原生实例输出不因 UNVEIL 修改。
- **对 wiki 的映射：** [UNVEIL 核心机制与源码时序](../../wiki/entities/paper-unveil-pcm-2611-25747.md)。

## 原文摘录 3：PCM-Bench 的实验口径

- PCM-Bench 在 **15 个测试集 / 31,175 张图像**上覆盖七项主任务；不同任务先计算各自指定指标再做平均，不能当成所有任务共用的 mIoU。
- 论文报告七任务均值：**UNVEIL 90.5**，SAM 3 frozen **46.5**，SAM 3 presence=1 **61.8**，SAM 3 full fine-tuning **81.6**，SAM 3 LoRA **82.6**，ConceptSeg-R1 **74.2**（依论文比较配置）。
- UNVEIL 单查询对存在概念接受率 **98.8%**、对不存在概念拒绝率 **84.7%**；联合查询两项分别 **94.3% / 97.5%**。拒绝评测与正样本概念图质量为不同指标。
- LVIS 留出 200 类的平均 IoU **0.841**；冻结 SAM 3 **0.831**。该实验不能用来声称所有新对象类都大幅提升。
- 论文报告 RTX 4080 SUPER 单查询约 **138 ms**，显存分配峰值单查询 **4.2 GiB** / 八提示 **6.6 GiB**；数值依硬件与任务设置。
- **对 wiki 的映射：** [UNVEIL 实验与局限](../../wiki/entities/paper-unveil-pcm-2611-25747.md)。

## 原文摘录 4：空结果的训练覆盖限制

- verified negatives 主要覆盖 Context 概念、LVIS 对象类及少量镜面图像；光照/结构的其余概念缺少充足的“真正不存在”训练负例，**不能将 presence 拒绝能力无条件推广到所有短语**。
- 目标是 2D 可提示概念图，不是直接输出三维地图、机器人动作或实时闭环控制。机器人应用需额外的时序、深度、几何及安全验证。
- 使用受访问许可约束的 SAM 3 checkpoint；UNVEIL 的附加模型权重与仓库已公开，但端到端运行仍需满足 SAM 3 基座许可及 CUDA 硬件条件。

## 对 wiki 的映射

- [UNVEIL：可提示概念图独立详情](../../wiki/entities/paper-unveil-pcm-2611-25747.md)
- [SAM 3：原生 Promptable Concept Segmentation](../../wiki/entities/paper-sam3.md)
