# SplatAnything: Gaussian Spacing Normalization for Feed-forward 3DGS across Arbitrary Scenes

> 来源归档（ingest · 论文原文）

- **标题：** SplatAnything: Gaussian Spacing Normalization for Feed-forward 3DGS across Arbitrary Scenes
- **类型：** paper
- **作者：** Hao Yu, Zhenwei Wang, Yongzhen Hu, Yuning Gong, Junta Wu, Xiaowei Zhou, Hujun Bao, Tengfei Wang, Chunchao Guo, Sida Peng
- **机构：** Zhejiang University；Tencent Hunyuan；Causa Robotics
- **论文：** <https://arxiv.org/html/2611.08264v1>（arXiv:2611.08264v1，2026-11-05）
- **PDF：** <https://arxiv.org/pdf/2611.08264>
- **主页：** <https://zju3dv.github.io/SplatAnything/>
- **源码：** <https://github.com/zju3dv/SplatAnything>，详见 [源码归档](../repos/splatanything.md)
- **入库日期：** 2026-11-10
- **核对范围：** arXiv v1 全文：§1–5 的背景、公式 (1)–(14)、架构与训练、实验和局限；附录 A.1–A.9 的数据、视角采样、评测口径、实现细节、补充结果、推理效率、消融及全景应用；交叉核对官方项目页、README、inference.py、GaussianWorldMirror、GeometryWorldMirror、GaussianSplatRenderer 与 scene_io。
- **通读：** 已完成
- **一句话说明：** GSNorm 不用场景平均径向距离，而用由深度和相机焦距决定的高斯相邻像素间距中位数归一化场景，减小高斯尺度监督在跨场景、跨相机配置时的歧义。

## 核心论文摘录

### 1. 关键问题：RDNorm 不等于高斯间距归一化（§1、§3.2–3.3）

- 逐像素生成 Gaussian 的尺度需要与相邻 Gaussian 中心距离匹配；既有 RDNorm 以到参考相机原点的平均径向距离归一化，无法锁定高斯密度。
- 针孔模型、局部深度近似常数时，水平/竖直相邻像素反投影间距约为 D/fx、D/fy；两者几何平均 d ≈ D/sqrt(fx·fy)。
- 论文 Figure 4 指出 RDNorm 使不同场景类别的间距中位数最多存在约 4.5 倍变化；尺度预测可能产生空洞（过小）或模糊重叠（过大）。
- **对 wiki 的映射：** [SplatAnything](../../wiki/entities/paper-splatanything.md) 的核心原理与误区。

### 2. GSNorm 公式（§3.3，式 8–11）

- 定义 s_g = median_{i,u∈Ω_i}[ D_i(u) / sqrt(fx_i·fy_i) ]；把深度、3D 点和相机平移统一除以 s_g，使代表性的高斯间距归一化为约 1。
- GSNorm 使用整个场景有效像素的稳健统计量，而不是对每个像素独立改变尺度，旋转与内参不需要随之缩放。
- **实现差异：** 官方推理源码 `src/models/models/rasterization.py` 使用 `spacing_target=0.0013227513`，按 `target / median_spacing` 乘到深度和相机平移；数值尺度与论文规范坐标的 `median=1` 表述不同，原理一致。不能把此坐标单位直接理解为真实米制。
- **对 wiki 的映射：** [SplatAnything](../../wiki/entities/paper-splatanything.md) 的方法与工程实践。

### 3. 模型数据流与训练（§3.4–3.5；附录 A.4）

- 输入 N 张重叠 RGB 图像，外加可选相机内外参；冻结的 WorldMirror-2 几何模型预测深度和相机，几何再经 GSNorm 转换。
- Gaussian backbone + GS head 每像素预测尺度、旋转、不透明度和 SH 颜色系数，以深度反投影确定高斯中心；多视图高斯经体素加权合并。
- 训练采用图像重建监督，论文介绍跨视图颜色重建和感知约束；官方公开仓明确是 inference-only release，不能据此宣称完整训练已开源。
- **对 wiki 的映射：** [SplatAnything](../../wiki/entities/paper-splatanything.md) 的流程图与源码时序图。

### 4. 评测（§4；附录 A.3、A.5–A.8）

- 跨场景基准包含 ScanNet、NRGBD、DL3DV、Tanks and Temples、Mip-NeRF 360、ETH3D、DTU，覆盖物体、室内、户外及相机/尺度变化。
- 设计了已知/未知相机、输入视图数量（1/4/8/16/32/64）以及超出输入视角范围的新视角评价，主要考察 LPIPS、DISTS 和 PSNR。
- 论文 Table 3 在无已知位姿、N=8 的相同设定下：GSNorm DISTS 0.196 / LPIPS 0.237，RDNorm 0.223 / 0.272（越低越好）。这属于论文报告的对照结果，不是本仓实测。
- **对 wiki 的映射：** [SplatAnything](../../wiki/entities/paper-splatanything.md) 的评测与对比。

### 5. 局限（§5；附录 A.6、A.9）

- 极少视角、几何困难或严重未观察区域时，冻结几何先验本身的错误会直接限制高斯结果；方法不是补全未知内容的生成式世界模型。
- 跨数据集新视角合成质量不能推导为机器人导航成功率、碰撞几何精度或物理可交互仿真已验证。
- **对 wiki 的映射：** [SplatAnything](../../wiki/entities/paper-splatanything.md) 的局限与研究选型。

## 相关一手资料

- [SplatAnything 官方项目页](../sites/splatanything.md)
- [SplatAnything 推理源码](../repos/splatanything.md)
- [InfiniSplat 上一代](https://github.com/zju3dv/InfiniSplat)
