# VISTA · 物体装拆状态跟踪基准

- **类型：** dataset
- **原文：** [Hugging Face 数据卡](https://huggingface.co/datasets/davidegarde/VISTA)
- **固定版本：** [d9e1190f4e1a3d8da158e1bc64e16befef0b4233](https://huggingface.co/datasets/davidegarde/VISTA/tree/d9e1190f4e1a3d8da158e1bc64e16befef0b4233)
- **数据卡许可证：** CC BY 4.0；复用含上游素材时保留各来源的署名并核对原条款。
- **论文：** https://arxiv.org/abs/2611.27398v1
- **代码：** https://github.com/davidegardenal/TIPO
- **核对范围：** 数据卡、公开文件树、验证集列表、全部序列元数据和摩卡壶事件 JSON；未下载约 10 GB 图像与掩码压缩包。
- **通读：** 已完成
- **核对日期：** 2026-11-27

50 段评测序列来自 Assembly101（17）、IKEA ASM（8）、自录（25）；26,470 帧以及 256 个事件。数据包分 RGB 帧、实例掩码、事件 JSON、验证划分与逐序列元数据。论文表 1 写 10–20 FPS，发布元数据的实际取值有 6、10、20 和约 29.61 FPS，公开运行应读取逐序列 FPS。50 段中 3 段还在第 0 帧之后提供对象提示。

摩卡壶样例按 `frameIdx` 升序是 108、148、230、443，前两次一拆二，后两次二合一。事件 JSON 是事件工件而不是逐帧真值本身，掩码另存；驱动工程应用时不要把某一帧的物体交叠直接当成装配完成。
