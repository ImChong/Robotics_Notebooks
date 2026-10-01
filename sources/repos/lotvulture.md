# Lot Vulture

> 来源归档

- **标题：** Lot Vulture
- **类型：** repo
- **链接：** https://github.com/lotvulture/lotvulture
- **容器镜像：** `ghcr.io/lotvulture/lotvulture:latest`
- **项目页：** https://www.lotvulture.com/ — [`lotvulture-com.md`](../sites/lotvulture-com.md)
- **Stars：** ~87（2026-10-01）
- **入库日期：** 2026-10-01
- **一句话说明：** 边缘端车位占用 CV：RTSP/快照相机 → `vulturevision`（C++ **ONNX Runtime**）→ Vue 3 仪表盘、历史 scrubber、MQTT/Webhook 告警。
- **代码：** https://github.com/lotvulture/lotvulture（**源码可得**，Lot Vulture Source License 1.0 / PolyForm Shield 衍生）
- **沉淀到 wiki：** [lotvulture](../../wiki/entities/lotvulture.md)

---

## 核心定位

**软件定义停车场**：在物业本地服务器或 Windows 服务上跑 Web 栈（默认 `:8000`），对接任意厂商 **RTSP / ONVIF / 静态快照 URL**，用多边形编辑器映射车位，输出实时占用图、历史时间轴与自动化告警。

### 技术栈（README）

| 层 | 说明 |
|----|------|
| 推理 | `vulturevision` — C++ **ONNX Runtime**；Community **CPU**；Commercial **CUDA / DirectML** |
| 前端 | **Vue 3 + Vuetify**，Vue-Konva 车位多边形编辑 |
| 部署 | Windows 安装包、Docker / Compose、本地 `docs/DEVELOPMENT.md` 源码构建 |
| 运行时 | **Python 3.12**（应用侧）；平台 Linux / Windows |

### 版本与许可分层

| 能力 | Community | Commercial |
|------|-----------|------------|
| Bundled 模型 | Standard（~89% 页内口径） | High-Accuracy |
| 硬件 | CPU 优化 | GPU + CPU |
| 告警 / 编辑器 / 时间轴 | ✅ | ✅ |
| 云遥测与持续再训练 | — | ✅ |

### 开发者 Cloud API（可选）

POST `https://api.lotvulture.com/v1/detect` — 上传图像 + ROI 多边形 JSON；按图计费（项目页 $0.02/image），适合无本地 infra 的集成方。

---

## 对 wiki 的映射

- [Lot Vulture](../../wiki/entities/lotvulture.md)
- 交叉：[ONNX Runtime](../../wiki/entities/onnxruntime.md)、[Eclipse Mosquitto / MQTT](../../wiki/entities/mosquitto.md)
