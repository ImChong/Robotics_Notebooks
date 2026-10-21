# HFHPE

> 来源归档

- **标题：** HFHPE: High-Angle Robust Fast Head Pose Estimation
- **类型：** repo / 头部姿态估计
- **机构 / 作者：** PINTO0309（Katsuya Hyodo）
- **链接：** https://github.com/PINTO0309/High-Angle_Robust_Fast_Head_Pose_Estimation
- **论文：** 未见独立论文链接；README 给出 Zenodo DOI 10.5281/zenodo.23252205
- **入库日期：** 2026-10-20
- **许可证：** MIT
- **一句话说明：** 将 DINOv3 ViT-L/16 教师模型蒸馏到轻量学生网络，以整头 crop 一次推理输出 0–360° yaw、−120° 至 +120° pitch 与 0–360° roll，并提供 ONNX/TFLite 部署权重。
- **开放状态：** 已开源；代码仓库为 MIT，README 链出 ONNX/TFLite 权重发布包。
- **沉淀到 wiki：** 是 → [HFHPE 实体页](../../wiki/entities/hfhpe.md)

---

## 方法与部署摘要

- **输入与范围：** 检测器先提供整头区域 crop；模型覆盖后脑视角、侧脸、较大俯仰与遮挡。
- **一体化推理：** 轻量 roll 分支估计图像平面旋转，模型在 ONNX 图内反旋转 crop，再由 yaw/pitch 主干回归方向；每个角度同时输出 biternion（cos/sin）与 von Mises 集中参数 κ。
- **蒸馏与骨干：** DINOv3 ViT-L/16（320×320）为教师；学生包括 ViT-T/16、PP-HGNetV2-B0 与 MBConv。
- **速度边界：** README 报告 64×64 整合模型在 i9-10900K、ONNX Runtime 1.22 CPU EP 上单帧约 1.4–2.5 ms（随骨干而异）。这是姿态网络时间；示例中的 DEIMv2 头部检测另约 24 ms，且可替换为其他检测器。
- **数据与评测：** README 描述合成头姿数据与真实自拍样本，并用其验证集报告角度误差；未用常见公开头姿 benchmark，因此与其他论文数字不宜直接横比。
- **边界：** README 记录极低照度下可因 yaw 线索不足而失效；头部定位错误、模糊和照明条件仍会影响最终估计。

## 关键入口

- [官方 README](https://github.com/PINTO0309/High-Angle_Robust_Fast_Head_Pose_Estimation#readme) — 方法、验证集结果、运行和边界说明
- [权重发布页](https://github.com/PINTO0309/High-Angle_Robust_Fast_Head_Pose_Estimation/releases/tag/weights) — ONNX/TFLite 模型文件
- [Zenodo DOI](https://doi.org/10.5281/zenodo.23252205) — 项目归档记录
