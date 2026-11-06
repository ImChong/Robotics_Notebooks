# Faster R-CNN 论文通读（目标检测方法页视角）

> 来源归档（原文通读 · 支撑 `wiki/methods/object-detection.md`）

- **标题：** Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks
- **作者：** Shaoqing Ren, Kaiming He, Ross Girshick, Jian Sun
- **类型：** paper / computer-vision / object-detection
- **原文：** <https://arxiv.org/abs/1506.01497>（2026-11 通过 arXiv PDF v3 获取全文）
- **一句话说明：** 用与检测网络共享卷积特征的区域提议网络（RPN）取代 Selective Search，使提议近乎零成本，奠定两阶段检测的端到端范式。
- **核对范围：** 通读摘要、§1 引言、§2 相关工作、§3（RPN、anchor、损失、训练、特征共享的三种方案与 4 步交替训练、实现细节）、§4 实验（VOC 2007/2012、消融、一阶段对比、COCO、COCO→VOC）与 §5 结论；各类别 AP 明细表与参考文献仅浏览。ILSVRC/COCO 2015 竞赛结果仅按论文叙述记录，未核对外部来源。
- **通读：** 已完成

## 核心摘录（均出自原文）

- **动机（§1）：** SPPnet / Fast R-CNN 共享卷积后，区域提议成为测试期瓶颈：Selective Search 约 2 s/图（CPU），EdgeBoxes 约 0.2 s/图。
- **RPN（§3.1）：** 在共享的最后一层卷积特征图上滑动 3×3 小网络，映射到 256-d（ZF）/512-d（VGG）后接两个并列 1×1 卷积：cls（2k 个 object/非 object 分数）与 reg（4k 个框坐标）；提议额外开销约 10 ms/图。
- **Anchor（§3.1.1）：** 每个位置 k=9 个参考框（3 尺度 128²/256²/512² × 3 长宽比 1:1/1:2/2:1），平移不变；输出层约 2.8×10⁴ 参数，比 MultiBox 的 6.1×10⁶ 少两个数量级；单尺度图像与单一尺寸滤波器即可处理多尺度，不需图像/滤波器金字塔。
- **损失（§3.1.2）：** 与 anchor 的 IoU 最高者或 IoU>0.7 为正样本，<0.3 为负样本；L = Ncls⁻¹ΣLcls + λNreg⁻¹Σp*·Lreg，Lreg 为 smooth L1，λ=10；框回归参数化 tx=(x−xa)/wa、tw=log(w/wa) 等。每图随机采 256 个 anchor，正负至多 1:1。
- **特征共享（§3.2）：** 三种方案：交替训练、近似联合训练（省 25–50% 训练时间，忽略提议框坐标的导数）、非近似联合训练（需可微 RoI warping）；实验采用 4 步交替训练。
- **实现细节（§3.3）：** 图像短边缩放到 600；总步长 16；跨图像边界的 anchor 训练时忽略（约 20000 个降到约 6000），测试时裁剪到图像内；提议按 cls 分数做 NMS（IoU 0.7），约剩 2000 个，再取 top-N。
- **VOC 结果（§4.1，表 2–4）：** Fast R-CNN+ZF：SS 58.7 / RPN 共享 59.9 mAP（仅 300 提议）；VGG-16：RPN 共享 69.9（07）、73.2（07+12）、78.8（COCO+07+12）；VOC 2012 test 70.4（07++12）、75.9（加 COCO）。
- **消融（§4.1）：** 去掉 NMS 取 6000 提议 55.2（对比有 NMS 300 个 56.8），NMS 不损精度；去掉 cls 在 100 提议时降至 44.6；去掉 reg 降至 52.1（说明高质量提议主要来自回归）；单一 anchor 降 3–4 点；λ 在 1–100 间仅影响约 1%。
- **耗时（表 5，K40 GPU）：** VGG：SS+Fast R-CNN 1830 ms（0.5 fps）；RPN+Fast R-CNN 198 ms（5 fps）；ZF：59 ms（17 fps）。
- **一阶段对比（§4.1，表 10）：** 以 ZF 为骨干，OverFeat 式一阶段系统 53.9 mAP，低于两阶段 58.7（−4.8），论文据此肯定「级联提议+区域分类」的作用。
- **COCO（§4.2，表 11）：** VGG-16，RPN 300 提议，COCO trainval 训练在 test-dev 上 42.7 mAP@0.5 / 21.9 mAP@[.5,.95]；论文述及换用 ResNet-101 后 val 上 48.4/27.2，并为 COCO 2015 检测赛冠军方案（He 等）的基础。
- **结论（§5）：** RPN 使提议几乎零成本；检测精度随更强骨干同步提升；数据规模（COCO 预训练）对 VOC 有显著增益（+5.6 mAP）。

## 对 wiki 的映射

- [`wiki/methods/object-detection.md`](../../wiki/methods/object-detection.md)：两阶段路线（RPN + RoI）、NMS、ResNet-FPN 骨干与延迟的取舍。
