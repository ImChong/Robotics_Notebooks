# VGG-T³: Offline Feed-Forward 3D Reconstruction at Scale

> 来源归档（ingest）

- **标题：** VGG-T³: Offline Feed-Forward 3D Reconstruction at Scale
- **缩写：** **VGG-T³** / Visual Geometry Grounded Test Time Training
- **类型：** paper / 3d-reconstruction / foundation-model / sfm / pose-estimation / test-time-training
- **arXiv：** <https://arxiv.org/abs/2602.23361>（PDF: <https://arxiv.org/pdf/2602.23361>）
- **会议：** CVPR 2026
- **项目页：** <https://research.nvidia.com/labs/dvl/projects/vgg-ttt/>
- **代码：** <https://github.com/nv-dvl/vgg-ttt>
- **权重：** <https://huggingface.co/nvidia/vgg-ttt>（NVIDIA OneWay Noncommercial License）
- **机构：** 英伟达（NVIDIA DVL）；多伦多大学（University of Toronto）；矢量研究所（Vector Institute）
- **作者：** Sven Elflein、Ruilong Li、Sérgio Agostinho、Zan Gojcic、Laura Leal-Taixé、Qunjie Zhou、Aljosa Osep
- **状态：** arXiv 预印本（2026-02-26，v1）；**已开源**（推理 + 评测 + 训练 harness；数据集实现待发布）
- **入库日期：** 2026-09-09
- **一句话说明：** 在 VGGT 骨干上将全局 softmax attention 替换为 **测试时训练（TTT）** 压缩的固定尺寸 MLP，使离线前馈 3D 重建对输入视图数 **线性扩展**；1k 图约 **54 s**（论文）/ **58 s**（项目页演示），相对 VGGT **~11.6×** 加速，点图误差仍优于其他线性时间基线，并支持冻结场景表示后的 **视觉定位查询**。

## 摘录 1：问题与核心洞察

- **瓶颈：** 离线前馈 3D 重建（如 VGGT）在全局 attention 层对输入图像数的计算与显存需求 **二次增长**，限制大规模图像集 / 长视频批处理。
- **根因（论文）：** 场景几何的 **变长 KV 空间表示** 迫使 softmax attention 随视图数膨胀。
- **解法：** 用 **测试时训练（TTT）** 把 KV 空间 **蒸馏进固定尺寸 MLP**；保留全局场景聚合能力，但复杂度对视图数 **线性**（接近在线模型）。
- **命名：** VGG-T³ = **V**isual **G**eometry **G**rounded **T**est **T**ime **T**raining；基于 [VGGT](https://github.com/facebookresearch/vggt) 并 **兼容其 API**。

**对 wiki 的映射：** 升格 [`wiki/entities/paper-vgg-ttt.md`](../../wiki/entities/paper-vgg-ttt.md)；交叉 [Glob3R](../../wiki/entities/paper-glob3r.md)、[LingBot-Map](../../wiki/methods/lingbot-map.md)、[State Estimation](../../wiki/concepts/state-estimation.md)。

## 摘录 2：方法（线性化全局 attention + TTT）

- **结构：** 保留 VGGT 的 frame / global 交替块；仅将 **全局 attention 块** 换为基于 TTT 的线性时间替代（项目页图示：左 VGGT softmax global attention → 右 TTT MLP 压缩 KV）。
- **初始化：** **线性化现有 VGGT checkpoint**——几乎全层加载预训练权重，**仅微调全局 attention 层**；作者强调 linearization 对性能关键。
- **TTT 机制：** 参考 LaCT / Test-Time Training Done Right；`vggttt/nets/ttt.py` 自 LaCT 改编（MIT）。
- **长度泛化：** 训练分布内序列 **1 步** optimizer 足够；对 **1k 图** 等 OOD 长度，增至 **2 步** 即可接近完美长度泛化（项目页 Fig (b) 橙线）。
- **查询模式（视觉定位）：** 场景批处理后 **冻结 TTT 优化得到的 MLP 权重**；对新查询图仅对 query 特征 $q_i$ 过冻结 MLP 读取场景表示、**不再更新** $\theta$，等效单图 transformer；项目页报 **~10 FPS** 实时定位。

**对 wiki 的映射：** 实体页「核心原理 / 流程总览 / 源码运行时序图」。

## 摘录 3：输出与接口

- **推理 API（与 VGGT 对齐）：**
  ```python
  from vggttt.nets.vggt.models.vggt import VGGT
  vggttt = VGGT.from_pretrained("nvidia/vgg-ttt").eval().cuda()
  preds = vggttt.infer(images)
  # pose [#N,4,4], intrinsics [#N,3,3], pts3d [#N,H,W,3], conf, depth
  ```
- **模型规模：** ViT 骨干，约 **1.19×10⁹** 参数；输入最大 **518×518** RGB；支持图像集与视频（按帧率拆帧）。
- **Demo：** `python vggttt/demo.py`（Viser + Gradio 交互重建与可视化）。

**对 wiki 的映射：** 实体页「工程实践」；[`sources/repos/vgg_ttt.md`](../repos/vgg_ttt.md)。

## 摘录 4：实验与可扩展性

- **点图评测（Table 1）：** DTU、ETH3D、NRGBD、7-Scenes 等；`vggttt/evaluation/pointmaps/eval.py`。
- **视觉定位（Table 5）：** 7-Scenes、Wayspots；`vggttt/evaluation/visloc/eval.py`。
- **可扩展性（Fig 4 / Table 4）：** 7-Scenes / NRGBD strided 上附加 support views **100 / 500 / 1000**（分布式最高 **2000**）；论文报 1k 图 **54 s**、相对 softmax attention 基线 **11.6×** 加速。
- **定性（项目页）：** 1k 图序列上 **VGGT ~11 min**、**TTT3R ~61 s**（重建不完整）、**VGG-T³ ~58 s**（完整重建；质量略低于 VGGT 但远快于 VGGT）。

**对 wiki 的映射：** 实体页评测表与结论。

## 摘录 5：开源边界与许可

| 项 | 结论 |
|----|------|
| **推理 + 评测** | **已开源** — [`nv-dvl/vgg-ttt`](https://github.com/nv-dvl/vgg-ttt) |
| **权重** | **已发布** — [nvidia/vgg-ttt](https://huggingface.co/nvidia/vgg-ttt) |
| **训练** | **部分**：`train.py` / `vggttt/trainer/` 已放出；**数据集实现与预处理缺失**，作者称正评估进一步开放可行性 |
| **许可** | 主体 **NVIDIA OneWay Noncommercial**（非商业科研/教育）；`vggttt/nets/vggt/` 沿用 **VGGT license**；`ttt.py` **MIT**（LaCT）；`evaluation/pointmaps/utils.py` **CC BY-NC-SA 4.0**（CUT3R） |
| **商业部署** | HF 卡与 README 均标明 **research & development only**；商用需另谈许可 |

**对 wiki 的映射：** [`sources/sites/nvidia-dvl-vgg-ttt.md`](../sites/nvidia-dvl-vgg-ttt.md)、实体页「工程实践 / 局限」。
