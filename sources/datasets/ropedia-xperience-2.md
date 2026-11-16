# Ropedia Xperience 2 数据集（Hugging Face）

> 来源归档（官方数据集卡）

- **数据集：** <https://huggingface.co/datasets/ropedia-ai/Ropedia-Xperience-2>
- **项目页：** <https://ropedia.com/xperience/2-research/>
- **论文：** <https://arxiv.org/abs/2611.17060>
- **官方源码：** <https://github.com/Ropedia/Ropedia-Xperience-2>
- **许可证：** CC BY-NC 4.0（署名、非商业；商用需单独授权）
- **核对日期：** 2026-11-13
- **核对范围：** 数据卡全文（规模、采集、结构、时间/坐标约定、质量、许可、隐私及下载示例）。
- **通读：** 已完成
- **发布状态：** 官方数据卡明确写明正在分批上传；首批列出 126 个场景，并逐场上传相关文件，后续扩展至 1,500+ 场景。数据查看器当时尚未就绪，不应将论文统计误作实时可下载数据量。

## 已公开的数据组织说明

- `metadata/scenes.parquet`、`captures.parquet`、`segments.parquet`、`files.parquet`：场景、录制、分段与文件索引。
- `scenes/<scene_id>/3d_scanning/`：3DGS PLY、网格 OBJ、COLMAP sparse、鱼眼图像和掩码。
- `scenes/<scene_id>/mapping/<capture_id>/`：头戴扫场关键帧、相机位姿、IMU、场景描述、点云等。
- `scenes/<scene_id>/mocap/<capture_id>/`：全身/手部 HDF5、设备轨迹、IMU、层次字幕、质量报告；标注和手部动作可能有估计误差。
- `scene_id` 为稳定 19 位标识，`capture_id` 为 UUID；帧编号、IMU 和轨迹文件存在纳秒/秒混合时间单位，位姿四元数也要区分 COLMAP 和 TUM 顺序。
- **重要边界：** 数据卡明确说明：首批扫描帧与头戴 SLAM 坐标系不同，**录制到扫描的配准结果目前未包含在首批数据文件内**；须用官方配准工具和对应原始导出处理，不能直接当作预对齐真值。
- 语言字幕由视觉语言模型自动生成，可能存在错误物体名、数量或动作；对隐私信息，官方提供 discussion 反馈删除路径。

## 访问示例（数据卡原文）

```python
from datasets import load_dataset
scenes = load_dataset("ropedia-ai/Ropedia-Xperience-2", "scenes", split="train")
print(scenes[0]["category"], scenes[0]["tasks"])
```

上述是元数据示例，是否成功取决于当时的上传进度与客户端环境；本次未实测下载全量文件。

## 对 wiki 的映射

- [Ropedia Xperience 2 实体页](../../wiki/entities/paper-ropedia-xperience-2.md)
- [论文归档](../papers/ropedia_xperience_2_arxiv_2611_17060.md)
