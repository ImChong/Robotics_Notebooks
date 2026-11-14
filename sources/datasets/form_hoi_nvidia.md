# FORM-HOI 数据集（NVIDIA Hugging Face）

> 数据集来源归档；根据公开数据卡与关联资料核对记录。

- **类型：** dataset / human-object-interaction / multiview / 4d-reconstruction
- **URL：** <https://huggingface.co/datasets/nvidia/form-hoi>
- **名称：** FORM-HOI（Foundry for Reconstruction from Multiview HOI）
- **所有者：** NVIDIA Corporation
- **数据创建日期 / 版本：** 2026-07 / 0.1.0
- **许可：** CC BY 4.0（支持商业与非商业使用）
- **配套项目与代码：**
  - 代码仓（重建与机器人 grounding 模块）：<https://github.com/nvidia-isaac/video_to_data>
  - 关联重建模型（[CARI4D](../../wiki/entities/paper-cari4d.md)）：<https://nvlabs.github.io/CARI4D/>（arXiv:2512.11988）
- **核对范围：** 2026-11-14 通读 Hugging Face `nvidia/form-hoi` 完整数据集卡（95 行），对照 `arXiv:2512.11988`（CARI4D 全文）核对其下游训练定位及数据边界；官方管线源码（`nvidia-isaac/video_to_data`）已另行检出并核对归档于 [video_to_data 官方管线来源](../repos/video_to_data_form_hoi.md)。
- **通读：** 已完成
- **一句话说明：** NVIDIA 开源的四视角标定 RGB-D 人–物交互多模态数据集，清洗发布集包含 4,135 段（约 29.144 小时）、61 个物体 ID 与 22 位参与者，提供 SOMA/MHR 人体姿态、SE(3) 物体位姿、度量 GLB 网格与逐区间质量核查标记。

## 核心技术要素摘录

1. **清洗发布集量化规模（不含 held-out 物体序列）：**
   - **Episodes：** 4,135 段；**总时长：** ~29.144 小时（每段只计一次，不随视角数乘 4）。
   - **个体覆盖：** 61 个唯一物体 ID；22 个唯一人员 ID。
   - **存储体积：** ~4.504 TB（十进制 TB，不含 release manifests）。
2. **单段（Episode）文件与模态契约：**
   - **4 路标定视觉流：** RGB（1536×1152，H.264 MP4）、深度图（768×576，HDF5）、人体掩码（1536×1152，黑白 HDF5）、物体掩码（1536×1152，黑白 HDF5）。
   - **标定参数：** `edex` JSON 格式存储每台相机的内外参。
   - **人体与物体 4D 重建标注：** SOMA 人体关节与体型（`soma_params.npz`）、MHR 人体参数（`mhr_params_mv.pt`）、逐帧 SE(3) 物体刚体变换（`poses.npy`，形状 `(N, 4, 4)`）、度量尺度带纹理物体网格（`output_aligned.glb`）、主轴旋转对称性（`output_symmetry.json`）及地面平面方程系数 `Ax+By+Cz+D=0`（`ground_plane.json`）。
3. **混合标注与质量评估（`failure_segments.json`）：**
   - 由自动化多视角重建管线生成伪真值，并结合人工复核标注不合格半开帧区间（half-open intervals）。
   - 质检包含三类信号：**Human QC**（渲染叠加人工目检判定的位姿不准区间）、**Chamfer distance**（重建几何与观测深度不一致区间，实践中常由深度噪声而非标注本身引起）、**Silhouette bounding box containment**（重投影轮廓与分割掩码框不一致区间）。数据卡明确提示：自动化指标标记不等于已确认的标注错误。
4. **用途与当前资料边界：**
   - 数据卡明确其适用于训练 CARI4D 一类 4D HOI 重建模型及机器人策略 grounding；完整技术报告处于 in progress 状态，源码集成在 `nvidia-isaac/video_to_data` 的 `reconstruction/` 与 `robotic_grounding/` 模块（与 CHORD 等工作共享 V2D 底座）。

## 对应 wiki 节点

- [FORM-HOI](../../wiki/entities/dataset-form-hoi.md)
