# Lot Vulture 官方项目页（lotvulture.com）

> 来源归档

- **标题：** Lot Vulture — Intelligent Parking Lot Management & Occupancy AI
- **类型：** site / project-page
- **URL：** <https://www.lotvulture.com/>
- **代码：** <https://github.com/lotvulture/lotvulture> — 归档见 [`sources/repos/lotvulture.md`](../repos/lotvulture.md)
- **Developer Cloud API：** <https://www.lotvulture.com/developer.html>（按图计费 REST，`/v1/detect`）
- **入库日期：** 2026-10-01
- **一句话说明：** 用现有 IP/RTSP 监控摄像头做车位级占用检测与运营分析；**本地边缘推理**，Raw 视频不出园区网。

## 开源核查（步骤 2.5，截至 2026-10-01）

| 核查项 | 结论 |
|--------|------|
| 项目页 Footer / 站点是否链 GitHub | **是** → `lotvulture/lotvulture` |
| 安装入口 | Windows `.exe` [Releases](https://github.com/lotvulture/lotvulture/releases)；Docker `ghcr.io/lotvulture/lotvulture:latest` |
| 许可证 | **Lot Vulture Source License 1.0**（PolyForm Shield 1.0.0 改编）；Community 可免费自托管；**商用高精度模型权重与 GPU 加速需 Commercial License** |
| 综合判定 | **源码可得（source-available）**：可运行、可改 Community 栈；**非** OSI 意义下宽松开源；高精度 ONNX 权重与云 attestation 资产单独授权 |

## 页面要点（2026-10-01）

- **价值主张：** 不铺地磁/地感，复用已有安防摄像头；宣称相对专用硬件 **>90% 成本下降**、Community 模型 **~89%**、Commercial **98%+** 精度（营销页口径）。
- **隐私：** 计数级检测、**不读车牌/人脸**；FAQ 强调 **100% 本地边缘**、原始 RTSP **不上传公有云**（许可心跳每 10 分钟，**14 天离线宽限期**）。
- **产品形态：** Community（CPU、标准模型、$0）；Commercial（GPU CUDA/DirectML、高精度模型、云遥测与持续再训练、优先支持）。
- **集成：** Webhook、**MQTT**（Home Assistant / 路侧屏）、邮件告警；Developer Cloud 按图 API 可选。

## 关联资料

- 仓库归档：[`sources/repos/lotvulture.md`](../repos/lotvulture.md)
- Wiki 实体：[`wiki/entities/lotvulture.md`](../../wiki/entities/lotvulture.md)
