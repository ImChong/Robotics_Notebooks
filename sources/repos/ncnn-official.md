# ncnn 官方仓库索引

> 来源归档（以 Tencent/ncnn GitHub README 叙述为准）

- **标题：** ncnn — high-performance neural network inference framework for mobile
- **类型：** 开源 C++ 推理框架（无第三方运行时依赖）
- **仓库：** https://github.com/Tencent/ncnn
- **入库日期：** 2026-06-25
- **一句话说明：** 腾讯开源的 **移动端/嵌入式高性能推理框架**：纯 C++、**无 BLAS 等第三方依赖**，深度优化 **ARM NEON** 与 **Vulkan GPU**；通过 **pnnx** 等工具从 **PyTorch / ONNX** 转换；广泛用于 **Android/iOS 感知**（YOLO 系、人脸检测等）；机器人语境下适合 **ARM 机载视觉、资源极紧的 CNN 推理**，与人形 **高频策略 C++ 环**（多走 ORT/TRT）形成分工。
- **沉淀到 wiki：** [ncnn](../../wiki/entities/ncnn.md)

---

## GitHub README 要点

1. **设计目标**：从设计之初面向 **手机端部署**；跨平台（Android、iOS、Linux、Windows、macOS 等）。
2. **性能**：官方称在移动端 CPU 上快于诸多开源框架；支持多核、big.LITTLE 调度、Vulkan 加速。
3. **模型格式**：`.param` + `.bin`；支持 **fp16、int8 量化**、自定义层。
4. **转换工具**：
   - **pnnx** — PyTorch / ONNX → ncnn（新主线）
   - _legacy_ 转换器支持 Caffe 等旧格式
5. **生产使用**：QQ、Qzone、微信、Pitu 等腾讯系应用。
6. **任务覆盖**：检测、分割、姿态等 CNN 工作负载为主；文档亦提及 broader than CNN-only。

---

## 与本仓库知识的关系

| 主题 | 关系 |
|------|------|
| [MNN](../../wiki/entities/mnn.md) | 同属中国移动端推理生态；MNN 偏阿里系全栈，ncnn 偏零依赖轻量 |
| [ONNX](../../wiki/entities/onnx.md) | pnnx 常见上游为 ONNX / PyTorch 导出 |
| [RF-DETR](../../wiki/entities/rf-detr.md) | 感知部署亦可走 ONNX→TRT；ncnn 为 ARM 视觉备选 |
| [object-detection 选型](../../wiki/queries/object-detection-model-selection.md) | 机载边缘 YOLO + 量化路径的对照选项 |

---

## 对 wiki 的映射

- 新建 **`wiki/entities/ncnn.md`**
- 纳入机载推理 runtime 对比页的 **延伸生态** 表

---

## 外部参考

- [Tencent/ncnn](https://github.com/Tencent/ncnn)
- [pnnx](https://github.com/pnnx/pnnx)
