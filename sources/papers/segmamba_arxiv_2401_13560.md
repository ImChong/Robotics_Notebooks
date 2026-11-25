# SegMamba: Long-range Sequential Modeling Mamba For 3D Medical Image Segmentation（基于长程序列建模的三维医学图像分割）

- **类型：** paper
- **论文：** https://arxiv.org/abs/2401.13560v4
- **HTML：** https://arxiv.org/html/2401.13560v4
- **PDF：** https://arxiv.org/pdf/2401.13560v4
- **作者：** Zhaohu Xing, Tian Ye, Yijun Yang, Guang Liu, Lei Zhu
- **版本：** v4，2024-09-15
- **代码：** https://github.com/ge-xing/SegMamba
- **收录日期：** 2026-11-24
- **核对范围：** v4 HTML 正文 §1–4、公式 (1)–(4)、数据集与表 1/2/5/8/9；另读官方 0_inference.py 与 model_segmamba/segmamba.py，区分论文和代码版本。
- **通读：** 已完成

## 核心摘录（意译）

- 编码器按尺度堆叠 TSMamba 块：GSC 提取局部空间信息，ToM 沿正向、反向和跨切片方向处理序列，结合残差 MLP；卷积解码器通过经 FUE 增强的跳跃连接获得多尺度特征。
- v4 Table 2：BraTS2023 平均 Dice 91.32%，平均 HD95 3.56；同设置 UX-Net 为 89.69%、4.81。评测采用作者随机划分，不能等同官方挑战排行榜。
- Table 9 对比高分辨率全局建模：同表自注意力配置发生显存溢出；这一结果不能外推到所有注意力实现或机器人部署。
- 作者 README 提供预处理、训练、推理和指标计算代码；快速测试入口为 `python 0_inference.py`，预处理先执行 `1_rename_mri_data.py`，再执行 `2_preprocessing_mri.py`。未执行程序；已阅读随机张量推理入口与模型前向调用链。
- 数据： [BraTS 2023 Adult Glioma](https://www.synapse.org/Synapse:syn51156910/wiki/622351) 需遵守数据访问条款；CRC-500 访问依作者说明，不声称直接公开下载。

## 对 wiki 的映射

- [视觉 Transformer 课程策展](../../wiki/entities/transformer-cv-curriculum.md)：第 7 章作业 6。
- [U-Mamba](../../wiki/entities/u-mamba.md)：独立的相关医学图像分割方法，不是 SegMamba 的同义词。
- [SSM](../../wiki/concepts/state-space-model-ssm.md)：序列状态建模先修。

- [SegMamba 独立论文详情](../../wiki/entities/paper-segmamba.md)：论文与实际源码双图。

源码核查：`GSC.forward` 使用两支相加（论文式 (2) 为乘法门控）；公开 `SegMamba.forward` 未显示 FUE 的显式调用，不假设代码完全实现 v4。尚未运行 GPU 推理。
