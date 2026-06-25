# ONNX Runtime 官方站点与文档索引

> 来源归档（以 onnxruntime.ai 官网与 GitHub README 叙述为准；Execution Provider 列表以目标版本文档为准）

- **标题：** ONNX Runtime — Accelerated Edge Machine Learning
- **类型：** 跨平台推理/训练引擎 + 多语言 API
- **主页：** https://onnxruntime.ai/
- **文档：** https://onnxruntime.ai/docs/
- **核心代码：** https://github.com/microsoft/onnxruntime
- **Generative AI 扩展：** https://onnxruntime.ai/docs/genai/（`onnxruntime-genai`）
- **入库日期：** 2026-06-25
- **一句话说明：** 微软主导的 **生产级 ONNX 推理与训练加速引擎**，支持 **Python / C++ / C# / Java / JavaScript** 等语言，覆盖 **Linux / Windows / macOS / iOS / Android / Web**；通过 **Execution Provider（EP）** 对接 CPU、CUDA、TensorRT、OpenVINO、CoreML、NNAPI 等后端；广泛用于 **Windows、Office、Azure、Bing** 及机器人 **C++ 机载策略推理**（如 Unitree G1 WBC、AMP_mjlab 部署链）。
- **沉淀到 wiki：** [ONNX Runtime](../../wiki/entities/onnxruntime.md)

---

## 首页要点（2026-06-25 抓取归纳）

1. **定位**：「Production-grade AI engine to speed up training and inferencing in your existing technology stack.」
2. **快速安装**：`pip install onnxruntime`；生成式场景另提供 `pip install onnxruntime-genai`。
3. **最小 Python 推理**：
   ```python
   import onnxruntime as ort
   session = ort.InferenceSession("model.onnx")
   outputs = session.run(None, {"input": input_tensor})
   ```
4. **能力轴**：
   - **Cross-Platform**：多 OS + 移动端 + 浏览器（ONNX Runtime Web / Mobile）。
   - **Performance**：针对延迟、吞吐、内存与二进制体积优化；可按用例进一步调优。
   - **Generative AI**：LLM 等生成模型本地/边端推理叙事。
5. **训练侧**：ONNX Runtime Training 支持大模型训练加速与 **on-device training** 叙事。

---

## 与 ONNX 格式的关系

- **ONNX**（[onnx.ai](https://onnx.ai/)）定义 **`.onnx` 文件与算子规范**。
- **ONNX Runtime** 是 **执行该格式的运行时** 之一（另有 TensorRT 直接 ingest ONNX、MNN 经 convert 等路径）。
- 机器人栈常见分工：**PyTorch/JAX 训练 → 导出 ONNX → ORT（CPU/GPU EP）或 ORT+TensorRT EP 上机**。

---

## Execution Provider（EP）概念（归纳）

ORT 通过 **SessionOptions** 注册 EP，按优先级调度算子：

| EP（示例） | 典型场景 |
|------------|----------|
| CPU | 通用回退、x86/ARM 机载 |
| CUDA | NVIDIA GPU 数据中心 / Jetson |
| TensorRT | NVIDIA 上进一步图优化与 INT8/FP16 |
| OpenVINO | Intel CPU/GPU/VPU |
| CoreML / NNAPI | iOS / Android 移动 |

> 具体 EP 可用性与算子覆盖须以 [官方 EP 文档](https://onnxruntime.ai/docs/execution-providers/) 与目标 `.onnx` 图为准。

---

## 与本仓库知识的关系

| 主题 | 关系 |
|------|------|
| [ONNX](../../wiki/entities/onnx.md) | 格式规范；ORT 是最常用的兼容 runtime |
| [wbc-fsm](../../sources/repos/wbc_fsm.md) | G1 **纯 C++ + ONNX Runtime 1.22** 部署 LAFAN1 WBC 策略 |
| [AMP_mjlab](../../wiki/entities/amp-mjlab.md) | 训练导出 ONNX → C++ ORT 推理 |
| [jackhan-feap-mujoco-deployment](../../wiki/entities/jackhan-feap-mujoco-deployment.md) | README 固定 `onnxruntime==1.19.2` 类版本提示 |
| [BotLab MotionCanvas](../../sources/sites/botlab_motioncanvas.md) | 浏览器端 **ONNX Runtime WASM/WebGPU** 编排 obs→policy |
| [Humanoid-GPT](../../wiki/entities/paper-humanoid-gpt.md) | 真机对比提及 ONNX + TensorRT 低延迟部署 |

---

## 对 wiki 的映射

- 新建 **`wiki/entities/onnxruntime.md`**：runtime 实体页（EP、语言绑定、机器人 C++ 部署）。
- 参与 **`wiki/comparisons/onnxruntime-vs-mnn-vs-tensorrt.md`** 选型对比。

---

## 外部参考（便于复核）

- [ONNX Runtime 官网](https://onnxruntime.ai/)
- [文档首页](https://onnxruntime.ai/docs/)
- [Execution Providers](https://onnxruntime.ai/docs/execution-providers/)
- [microsoft/onnxruntime（GitHub）](https://github.com/microsoft/onnxruntime)
- [ONNX Runtime Web](https://onnxruntime.ai/docs/tutorials/web/)
