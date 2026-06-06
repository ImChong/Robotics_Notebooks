# 经典视觉骨干与实时检测文献簇（ResNet + YOLO 及相关）

> 来源归档（ingest 合集）

- **类型：** paper-collection / computer-vision / perception
- **入库日期：** 2026-06-06
- **最后更新：** 2026-06-06
- **一句话说明：** 围绕 **ResNet（表征深度）** 与 **YOLO v1（检测实时性）** 的两篇奠基论文，及其在 **两阶段检测、骨干预训练、机器人机载感知** 中的上下游资料索引。

## 主论文（本批 ingest 核心）

| 论文 | arXiv | 角色 |
|------|-------|------|
| Deep Residual Learning for Image Recognition | [1512.03385](https://arxiv.org/abs/1512.03385) | 极深 CNN 骨干；COCO/检测特征提取底座 |
| You Only Look Once | [1506.02640](https://arxiv.org/abs/1506.02640) | 单次回归实时检测；机器人视觉常用谱系起点 |

详细摘录见：
- [resnet_arxiv_1512_03385.md](./resnet_arxiv_1512_03385.md)
- [yolo_arxiv_1506_02640.md](./yolo_arxiv_1506_02640.md)

## 检测范式对照（两阶段 vs 单阶段）

| 路线 | 代表 | 特点 | 机器人语境 |
|------|------|------|------------|
| **区域提议 + 分类** | R-CNN → Fast/Faster R-CNN | 精度高、管线复杂、延迟大 | 离线标注、高精度抓取位姿 |
| **单次回归** | YOLO / SSD | 端到端、FPS 高、定位略弱 | **机载实时**（足球、导航、避障） |
| **密集预测 + NMS** | RetinaNet 等 | 单阶段 + focal loss 平衡类别 | 中等算力边缘设备 |

## 骨干与预训练链条

```text
ImageNet 分类预训练（ResNet / GoogLeNet）
        ↓ 微调分辨率 & 任务头
检测 / 分割 / 机器人策略视觉编码器（DINO、CLIP、ResNet、ViT…）
```

- **ResNet** 解决 **「能训多深」**；**YOLO** 解决 **「能跑多快」**；现代栈常二者结合（如 YOLO 系列换更强 CSP/Darknet 骨干，或检测器 + ResNet-FPN）。
- 机器人 wiki 中 **ResNet** 亦见于触觉 CNN、Doorman 视觉学生、NMR 1D-ResNet 等变体语境——见各实体页。

## 对 wiki 的映射

- [`wiki/entities/paper-resnet-deep-residual-learning.md`](../../wiki/entities/paper-resnet-deep-residual-learning.md)
- [`wiki/entities/paper-yolo-unified-realtime-detection.md`](../../wiki/entities/paper-yolo-unified-realtime-detection.md)
- [`wiki/concepts/vision-backbones.md`](../../wiki/concepts/vision-backbones.md)
- [`wiki/methods/object-detection.md`](../../wiki/methods/object-detection.md)
- [`wiki/concepts/deep-learning-foundations.md`](../../wiki/concepts/deep-learning-foundations.md)

## 当前提炼状态

- [x] 文献簇索引与范式对照
- [x] wiki 页面映射
