# FORM-HOI 数据集（NVIDIA Hugging Face）

> 数据集来源归档；根据公开数据卡记录。核对日期：2026-10-06。

- **类型：** dataset
- **URL：** https://huggingface.co/datasets/nvidia/form-hoi
- **名称：** FORM-HOI（Foundry for Reconstruction from Multiview HOI）
- **所有者：** NVIDIA Corporation
- **数据创建日期 / 版本：** 2026-07 / 0.1.0
- **许可：** CC BY 4.0
- **清洗发布集规模：** 4,135 episodes；约 29.144 小时；61 个物体 ID；22 个人员 ID；约 4.504 TB。统计排除 held-out object sequences；总时长按每段 episode 计一次。
- **每段数据：** 四路标定 RGB-D 与人/物掩码；SOMA/MHR 人体姿态；逐帧 SE(3) 对象 pose；度量尺度 GLB 对象 mesh；ground plane 与质量审核段。
- **数据格式：** 4× RGB 1536×1152 H.264 MP4；4× depth 768×576 HDF5；掩码 HDF5；pose N×4×4 数组。
- **标注：** 自动管线生成并配合人工审查；quality flags 是潜在问题提示，指标异常不总是确认为标注错误。
- **用途：** 数据卡明确提到 HOI 重建训练（如 CARI4D）及机器人策略 grounding。
- **处理代码：** https://github.com/nvidia-isaac/video_to_data
- **实体页：** [dataset-form-hoi.md](../../wiki/entities/dataset-form-hoi.md)
