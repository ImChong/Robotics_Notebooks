# SAM3DBody-cpp（SAM 3D Body 纯 C++ 推理与 BVH 导出）

> 来源归档

- **标题：** SAM3DBody-cpp — Standalone C++ inference for SAM-3D-Body
- **类型：** repo
- **作者：** AmmarkoV 等社区实现
- **代码：** <https://github.com/AmmarkoV/SAM3DBody-cpp>
- **上游训练/导出工具：** <https://github.com/AmmarkoV/Fast-SAM-3D-Body>
- **官方参考实现：** <https://github.com/facebookresearch/sam-3d-body>
- **ONNX 模型：** <https://huggingface.co/AmmarkoV/SAM3DBody-cpp-onnx-models>
- **入库日期：** 2026-05-30
- **一句话说明：** 运行时零 Python 依赖（ONNX Runtime + ggml）；单目 RGB/视频多人 MHR 姿态、可选 18439 顶点网格与 70 关键点；支持多人体 BVH 动捕导出、Butterworth/四元数滤波与离线五遍精修可执行文件。
- **沉淀到 wiki：** [SAM3DBody-cpp](../../wiki/entities/sam3dbody-cpp.md)

---

## 技术栈快照

| 模块 | 实现 |
|------|------|
| 检测 | YOLO11m-pose（`yolo.onnx`） |
| 编码 | DINOv3-ViT-H/14+ backbone（CUDA BF16 或 CPU fp32 变体） |
| 解码 | 6-layer PromptableDecoder |
| 参数头 | `pipeline.gguf`（MHR + camera，ggml CPU） |
| 网格 | 原生 C LBS（`body_model.lbs`），等价 Python `mhr_forward` |
| 输出 | 519 维姿态参数、相机平移、70×3D 关键点、可选顶点 |

## 典型用途

- 实时/准实时 **单相机全身跟踪** → CSV / BVH → Blender（`blender_bvh_plugin.py` + MakeHuman）
- 嵌入式或 **无 PyTorch** 部署（ROS demo、ctypes 轻量 Python 前端）
- 视频 **`--bvh`** 多人体动捕（bbox IoU 跟踪 + 骨长自适应 OFFSET）

## 对 wiki 的映射

- 主实体页：**`wiki/entities/sam3dbody-cpp.md`**
- 方法背景：**`wiki/entities/sam-3d-body.md`**
- 论文：**`sources/papers/sam_3d_body_arxiv_2602_15989.md`**
