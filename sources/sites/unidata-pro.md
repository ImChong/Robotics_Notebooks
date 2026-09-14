# Unidata（unidata.pro）

> 来源归档（site / data vendor）

- **名称：** Unidata
- **类型：** 商业 egocentric 数据采集与数据集供应商
- **官网：** <https://unidata.pro/>
- **总部：** Meydan Grandstand, 6th floor, Meydan Road, Nad Al Sheba, Dubai, UAE
- **入库日期：** 2026-09-14
- **一句话说明：** 以 **Pico 4 Ultra + Motion Trackers**（及可选 ZED 腕/头戴多相机）做规模化第一人称视频与全身/手部姿态采集，并提供商业数据集与采集服务。

## 开源状态（步骤 2.5）

- **采集软件栈：** **确认未开源** — 博客描述自研 **PICO Data collection service**（头显）与 **Orin Data collection service**（多相机 rig），无公开 GitHub / Hugging Face 链接。
- **数据集：** **商业可购** — [Egocentric Video Dataset](https://unidata.pro/)（**4,050 h** / 13 场景）、[Robotic Household Activities Dataset](https://unidata.pro/)（**1,000 h** 清洁/叠衣/洗碗）；各数据集页提供样例下载，非全量开放。
- **硬件：** 依赖 **PICO SDK**（`PXR_CameraImage` 等）与 **Stereolabs ZED SDK**（多相机 SVO2）；均为厂商 SDK，非 Unidata 自有开源。

## 核心产品 / 服务

| 产品 | 规模（截至 2026-08 博客） | 采集 rig |
|------|---------------------------|----------|
| Egocentric Video Dataset | **4,050 h** / **13** 日常活动场景 | 头显-only **2,321 h**；ZED+Pico **1,729 h** |
| Robotic Household Activities Dataset | **1,000 h** | 清洁、叠衣、洗碗等家务 |

## 对 wiki 的映射

- [`wiki/entities/pico-4-ultra-egocentric-capture.md`](../../wiki/entities/pico-4-ultra-egocentric-capture.md) — Pico 4 Ultra 采集规格、双 rig 工作流与设备对照
- [`sources/blogs/unidata_pico_4_ultra_egocentric_data_collection.md`](../blogs/unidata_pico_4_ultra_egocentric_data_collection.md) — 博客全文摘录
