# Dynamic SpectraFormer（arXiv:2608.18662）

> 来源归档（ingest）

- **标题：** Dynamic SpectraFormer for Ultra-High-Definition Underwater Image Enhancement
- **类型：** paper / underwater-robotics / image-enhancement / frequency-transformer
- **arXiv abs：** <https://arxiv.org/abs/2608.18662>
- **PDF：** <https://arxiv.org/pdf/2608.18662>
- **代码：** <https://github.com/arifence2024/DynamicSpectraFormer>（归档见 [`sources/repos/dynamic_spectraformer.md`](../repos/dynamic_spectraformer.md)）
- **机构：** 东京理科大学 Ishikawa Vision Lab；东京工业大学（Tokyo Tech）
- **作者：** Zhiqiang Hu、Shouren Huang、Masatoshi Ishikawa、Tao Yu
- **发表 / 上传：** 2026-08-21（arXiv v1）
- **入库日期：** 2026-08-21
- **索引来源：** [具身智能小站 8 篇综述](../blogs/wechat_embodied_station_8_papers_world_model_memory_2026-08-21.md)（<https://mp.weixin.qq.com/s/30hu9SRxbRNXJcGLnNwl_g>）

## 开源状态（步骤 2.5，2026-08-21）

- **待发布：** 论文摘要写 code available；GitHub 仓 **仅标题 README**，无训练/推理脚本或权重。
- **无独立项目页** — GitHub 为唯一入口。
- **结论：** **宣称可用 / 待核实**；截至入库日 **不可运行复现**。

## 摘录 1：退化机理

- 水下折射/吸收 → 色偏/雾化（**低频**）+ 边缘纹理退化（**高频**）；纯空间域难兼顾。

## 摘录 2：方法

- **频域增强**：超高清 **稀疏频谱注意** 建模长程依赖；**动态频谱权重层（DSWG）** 自适应强调关键频带、抑制次要频带。
- 多尺度 U-Net 骨干，面向 AUV/ROV 超高分辨率视觉。

## 摘录 3：评测

- 多组消融 + 多个水下图像增强基准验证有效性。

**对 wiki 的映射：** [`wiki/entities/paper-dynamic-spectraformer.md`](../../wiki/entities/paper-dynamic-spectraformer.md)。

## 当前提炼状态

- [x] GitHub 占位仓核查
- [x] 升格 `wiki/entities/paper-dynamic-spectraformer.md`
