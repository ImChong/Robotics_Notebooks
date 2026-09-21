# Offroad Global Nav（Hugging Face 数据集）

> 来源归档（dataset）

- **标题：** Offroad-global-nav Geospatial Dataset
- **类型：** dataset / geospatial / off-road-navigation / traversability / satellite / lidar
- **Hugging Face：** <https://huggingface.co/datasets/anony-008/offroad-global-nav>
- **组织：** `anony-008`（双盲投稿匿名账号；论文作者为 Texas A&M Unmanned Systems Lab + DEVCOM ARL）
- **论文：** <https://arxiv.org/abs/2607.23743>
- **许可：** CC BY-NC-4.0
- **入库日期：** 2026-09-21
- **一句话说明：** 面向公里级越野全局规划的多模态地理数据集：共配准卫星 GeoTIFF、航拍 LiDAR LAZ、OSM 矢量与人类 GPS（KML），覆盖美国多种非结构化地形。

## 规模（论文 §III / HF 卡片）

| 字段 | 数值 |
|------|------|
| 场景（论文口径） | **299** 处地理多样地点（HF 写还将追加） |
| 覆盖面积 | **~1,244 km²** |
| 人类驾驶 | **~1,130 km** GPS 轨迹 |
| HF 体积 | **~29.9 GB**（截至 2026-09-21 API） |
| HF rows | 卡片约 **201** 行（少于论文 299；部分场景仍在上传） |
| 许可 | **CC BY-NC-4.0**（非商业） |

## 模态与磁盘布局

每场景共配准、同一 USGS LPC 瓦片 ID 贯穿四类产物：

| 目录 | 格式 | 内容 |
|------|------|------|
| `data/geotiff/` | GeoTIFF `.tif` | 30 cm/px RGB 卫星（ArcGIS），植被 / 步道 / 水体外观 |
| `data/pointcloud/` | LASzip `.laz` | 航拍 LiDAR 5–27 pts/m²；高程、法向坡度、强度 |
| `data/OSM/` | GeoPackage `.gpkg` | OSM 公路、步道、水系（论文正文写 OSM XML；HF 发布为 GPKG） |
| `data/trajectories/` | KML `.kml` | 人类驾驶 GPS，作可通行隐式监督与路径偏好真值 |

文件名沿 USGS LPC 工程（如 `USGS_LPC_AZ_Coconino_2019_…`、`CA_SaltonSea`、`AR_Eastern`、`WY_North_Converse`、`HI_NOAAMauiOahu` 等），覆盖沙漠、草原、森林、山地、采石场。

## 加载入口

```python
from datasets import load_dataset

dataset = load_dataset("anony-008/offroad-global-nav")
sample = dataset["data"][0]
```

HF 标注 `format: imagefolder`、`modality: geospatial`。完整训练管线（栅格化 \(G_{\text{res}}\)、DINOv3-SAT 编码、nnPU 损失）**不在**本数据集仓内。

## 使用边界

- **非商业：** CC BY-NC-4.0，商用需另谈许可或改用公开 USGS/OSM/ArcGIS 原源并自建对齐。
- **不是局部感知集：** 无车载相机/雷达序列；对标 RUGD / RELLIS-3D / TartanDrive 的 onboard 任务需另找数据。
- **轨迹 ≠ 不可通行补集：** 未驶过像素是 unlabeled，不能当负样本（论文 nnPU 的核心假设）。

## 关联资料

- 论文：[`sources/papers/offroad_global_nav_arxiv_2607_23743.md`](../papers/offroad_global_nav_arxiv_2607_23743.md)
- Wiki：[`wiki/entities/paper-offroad-global-nav.md`](../../wiki/entities/paper-offroad-global-nav.md)
