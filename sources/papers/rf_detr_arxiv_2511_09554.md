# RF-DETR：面向实时检测 Transformer 的神经架构搜索（arXiv:2511.09554）

> 论文来源归档（ingest）

- **标题：** RF-DETR: Neural Architecture Search for Real-Time Detection Transformers
- **作者：** Isaac Robinson, Peter Robicheaux, Matvei Popov（Roboflow）；Deva Ramanan, Neehar Peri（CMU）
- **类型：** paper / computer-vision / object-detection / instance-segmentation / NAS
- **arXiv：** <https://arxiv.org/abs/2511.09554> · PDF：<https://arxiv.org/pdf/2511.09554.pdf>
- **会议：** ICLR 2026
- **代码：** <https://github.com/roboflow/rf-detr>
- **文档：** <https://rfdetr.roboflow.com/latest/>
- **入库日期：** 2026-06-22
- **一句话说明：** 以 **DINOv2 ViT 骨干 + 端到端 weight-sharing NAS** 现代化 specialist DETR：单次训练后在验证集 grid search 即可在 **精度–延迟 Pareto 前沿** 上选点，无需为每个子网重训；COCO 上 **RF-DETR-2XL 首破 60 AP**，RF100-VL 上显著优于 YOLO 系与多数实时 DETR。

## 核心摘录（面向 wiki 编译）

### 1) 问题：COCO 过拟合 vs 开放词汇 VLM 太慢

- **要点：** 开放词汇检测器（GroundingDINO、YOLO-World）零样本强，但 **域外类别/模态** 泛化差；微调 VLM 精度升但 **文本编码器拖慢推理**。专用实时检测器（D-FINE、RT-DETR）快，但 **在 RF100-VL 等真实分布上弱于微调 VLM**。RF-DETR 目标：**互联网规模预训练 + 实时架构 + 可针对目标域/硬件选点**。
- **对 wiki 的映射：** [`wiki/entities/rf-detr.md`](../../wiki/entities/rf-detr.md)、[`wiki/methods/object-detection.md`](../../wiki/methods/object-detection.md)

### 2) 架构：DINOv2 骨干 + LW-DETR 现代化 + 轻量分割头

- **要点：** 以 **LW-DETR** 为基，换 **DINOv2** 预训练 ViT（12 层 vs CAEv2 10 层）；**窗口/非窗口 attention 交错**；多尺度 projector 用 **LayerNorm** 替代 BatchNorm 以支持梯度累积。分割版 **RF-DETR-Seg** 在 encoder 输出上双线性插值 + 轻量 pixel embedding，query token 与 pixel map 点积得 mask；Objects365 上用 **SAM2 伪标签** 预训练分割头。
- **对 wiki 的映射：** [`wiki/entities/rf-detr.md`](../../wiki/entities/rf-detr.md)、[`wiki/concepts/vision-backbones.md`](../../wiki/concepts/vision-backbones.md)

### 3) 端到端 weight-sharing NAS（OFA 风格）

- **要点：** 训练时每步 **均匀采样** 子网配置并更新共享权重；搜索维度：**patch size、decoder 层数、query 数、输入分辨率、窗口 attention 块数**。推理时在验证集 **grid search** 选 Pareto 点，**无需对子网再 fine-tune**（COCO 上几乎无增益；RF100-VL 小数据集可选再训）。子网在训练中未见过时仍表现良好（Appendix F）。
- **对 wiki 的映射：** [`wiki/entities/rf-detr.md`](../../wiki/entities/rf-detr.md)

### 4) Scheduler-free 训练与域迁移

- **要点：** 反对 cosine LR / 复杂 augmentation schedule 对 **固定优化 horizon** 的隐含假设；仅 **水平翻转 + random crop**；**batch 级 resize** 减少 padding 浪费。DINOv2 初始化对小数据集迁移关键；NAS 的「architecture augmentation」同时作正则。
- **对 wiki 的映射：** [`wiki/entities/rf-detr.md`](../../wiki/entities/rf-detr.md)、[`wiki/queries/object-detection-model-selection.md`](../../wiki/queries/object-detection-model-selection.md)

### 5) 基准与延迟评测标准化

- **要点：** COCO detection/segmentation + **RF100-VL**（100 数据集平均）。**RF-DETR-N** 比 D-FINE-N **+5.3 AP** 同级延迟；**RF-DETR-2XL 60.1 AP**（T4 TensorRT FP16, 17.2 ms）为首个实时 **>60 AP** 检测器。提出延迟评测：**forward 间 buffer 200ms** 缓解 GPU 降频；**精度与延迟须同一模型 artifact**（FP16 量化 naive 可致 AP≈0）。
- **对 wiki 的映射：** [`wiki/entities/rf-detr.md`](../../wiki/entities/rf-detr.md)、[`wiki/methods/object-detection.md`](../../wiki/methods/object-detection.md)

## 关键数值（COCO val，T4 TensorRT FP16）

| 型号 | AP | AP50 | 延迟 (ms) | 参数量 |
|------|-----|------|-----------|--------|
| RF-DETR-N | 48.0 | 67.0 | 2.3 | 30.5M |
| RF-DETR-S | 52.9 | 71.9 | 3.5 | 32.1M |
| RF-DETR-M | 54.7 | 73.5 | 4.4 | 33.7M |
| RF-DETR-2XL | **60.1** | 78.5 | 17.2 | 126.9M |

RF100-VL：RF-DETR-2XL **63.2 AP50:95 平均**，快于 GroundingDINO-T **约 20×**。

## 相关资料索引

| 资料 | 关系 |
|------|------|
| [LW-DETR](https://arxiv.org/abs/2406.08460) | 直接前身架构 |
| [RT-DETR](https://arxiv.org/abs/2304.08069) | 实时 DETR 对照 |
| [D-FINE](https://arxiv.org/abs/2404.00999) | 同级实时 specialist 对照 |
| [DINOv2](https://arxiv.org/abs/2304.07193) | 预训练 ViT 骨干 |
| [RF100-VL / Roboflow100-VL](https://arxiv.org/abs/2503.07735) | 域迁移 benchmark |
| [rf-detr GitHub](../../sources/repos/rf_detr.md) | 官方实现与 `pip install rfdetr` |
| [RF-DETR 文档站](../../sources/sites/rfdetr-docs.md) | 安装、微调、部署教程 |

## 当前提炼状态

- [x] 要点摘录与 wiki 映射
- [x] 与机器人实时感知选型交叉引用
