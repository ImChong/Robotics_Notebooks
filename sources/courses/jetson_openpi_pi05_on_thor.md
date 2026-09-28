# Jetson AI Lab：OpenPi π₀.₅ on Jetson Thor

> 来源归档

- **标题：** OpenPi π₀.₅ on Jetson Thor
- **类型：** course（NVIDIA 官方边缘 VLA 部署教程）
- **来源：** Jetson AI Lab
- **链接：** https://www.jetson-ai-lab.com/tutorials/openpi_on_thor/
- **作者：** Aditya Sahu、Anqi Liu
- **入库日期：** 2026-09-28
- **代码：** https://github.com/Physical-Intelligence/openpi（**已开源**；Thor 部署脚本与 TRT 补丁由教程 `download.sh` 注入，**不在 upstream 主仓**）
- **部署脚本：** https://www.jetson-ai-lab.com/code-samples/openpi_on_thor/download.sh
- **一句话说明：** 在 Jetson AGX Thor 上将 Physical Intelligence **π₀.₅** VLA 走通 **JAX → PyTorch → ONNX（ModelOpt FP8/NVFP4）→ TensorRT → 推理 / WebSocket serve** 全链路，并给出 `pi05_libero` 上 **~49 ms** 级延迟基准。
- **沉淀到 wiki：** 是 → [`wiki/entities/jetson-openpi-pi05-on-thor.md`](../../wiki/entities/jetson-openpi-pi05-on-thor.md)

---

## 开源边界（步骤 2.5）

| 项 | 结论 |
|----|------|
| **OpenPi 权重** | GCS `gs://openpi-assets/checkpoints/` 公开下载，无需 auth |
| **openpi 仓库** | GitHub **Physical-Intelligence/openpi** 已开源；教程 **pin** `15a9616a00943ada6c20a0f158e3adb39df2ccac`（2026-06-16，`update output objects to support batching`） |
| **Thor 专用层** | `deployment_scripts/`（Dockerfile、`pytorch_to_onnx.py`、`build_engine.sh`、`pi05_inference.py` 等）+ 4 处 upstream 补丁 — 仅通过 Jetson AI Lab **`download.sh`** 获取 |
| **JetPack / 容器** | 基镜像 `nvcr.io/nvidia/pytorch:26.05-py3`；Jetson pip 索引 `https://pypi.jetson-ai-lab.io/sbsa/cu130` |

---

## 教程结构（步骤索引）

| 步骤 | 主题 |
|------|------|
| 前置 | Thor 硬件、JetPack **7.2**、CUDA **13+**、Docker **28.x**、NVIDIA Container Toolkit **1.18+** |
| Step 1 | `nvpmodel -m 0`、`jetson_clocks`（MAXN）；JP 7.0 GA 可选 GPU railgating 关闭 |
| Step 2 | clone openpi + submodules；checkout pin commit；`download.sh` 注入 Thor 脚本与补丁 |
| Step 3 | `deployment_scripts/thor.Dockerfile` 构建镜像 `openpi-pi0.5:l4t-jp7.2`（首建约 15–20 min） |
| Step 4 | nvidia runtime Docker，挂载 workspace 与 `~/.cache/openpi`、`~/.cache/huggingface` |
| Step 5 | `PYTHONPATH`；`CONFIG_NAME`（`pi05_libero` / `pi05_droid` / `pi05_aloha`）；复制 `transformers_replace` 补丁 |
| Step 6 | 自动下载 JAX checkpoint |
| Step 7 | `convert_jax_model_to_pytorch.py` → SafeTensors（约 5–10 min） |
| Step 8 | （可选）PyTorch BF16 推理冒烟 |
| Step 9 | `pytorch_to_onnx.py` — FP8 + LLM **NVFP4** + attention matmul QDQ |
| Step 10 | `build_engine.sh` + `trtexec` 编译 engine（约 10–30 min；固定 language len **208** tokens） |
| Step 11 | TensorRT 推理；默认 runtime hooks（tokenize cache、fast infer、CUDA graph） |
| Step 12 | （可选）`--inference-mode compare`：cosine ~0.99、~2.7× speedup |
| Step 13 | （可选）`serve_policy.py --use-tensorrt` WebSocket **:8000** |

---

## 核心摘录

### 1) 管线与定位

π₀.₅ 为 **flow-matching VLA**（10k+ h 机器人数据预训练）；Thor 提供 Blackwell 级 GPU + 最高 **128GB** 统一内存，教程目标是在板端 **闭环控制速率** 下跑通多模态管线。

```
JAX Checkpoint ──► PyTorch ──► ONNX (FP8 + NVFP4) ──► TensorRT Engine ──► Inference
```

### 2) 性能（AGX Thor DevKit，JetPack 7.2，MAXN，`pi05_libero`，action horizon 10）

| 推理后端 | Total (ms) | Model (ms) | Speedup |
|----------|------------|------------|---------|
| PyTorch BF16 | ~132 | ~128 | 1.0× |
| TensorRT FP8 | ~54 | ~53 | 2.4× |
| TensorRT FP8 + NVFP4 | ~49 | ~48 | ~2.7× |

权重体积 **~6 GB+**；推荐 NVMe SSD。

### 3) 量化与精度

- **不支持纯 FP16 导出**：π₀.₅ 原生 **BF16**（8-bit exponent）；FP16 动态范围不足，Gemma attention 在 denoising loop 中易溢出。
- **NVFP4**：最快路径；`compare` 模式 overall cosine **~0.994**（随机 noise 略有波动；可 `--golden-noise-path` 固定）。
- **FP8-only**（无 NVFP4）：更稳 cosine **≈0.9995**，latency ~53 ms。

### 4) 生产 serve

扩展 upstream `scripts/serve_policy.py`：`--use-tensorrt`、`--tensorrt-engine`；客户端仍用 **`openpi_client.websocket_client_policy`**，与 OpenPI 生态一致。

### 5) 致谢与参考

TensorRT 优化部分受 **FlashRT** 社区 Jetson Thor 结果启发。

---

## 对 wiki 的映射

- [jetson-openpi-pi05-on-thor](../../wiki/entities/jetson-openpi-pi05-on-thor.md) — 本教程知识页
- [Jetson AI Lab](../../wiki/entities/jetson-ai-lab.md) — 教程 hub
- [openpi 仓库归档](../repos/openpi.md)
- [π₀ 策略方法页](../../wiki/methods/π0-policy.md)
- [APXInf](../../wiki/entities/apxinf.md) — 第三方专用 VLA 引擎路线（与官方 TRT 教程互补对照）
- [VLA 真机部署指南](../../wiki/queries/vla-deployment-guide.md)
