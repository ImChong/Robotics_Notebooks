# InstantHMR｜官方代码与 README（2026-11-11 复核）

> 来源归档：官方项目、训练文档及推理源码；替换 2026-09-28 的早期仓库快照。

- **标题：** InstantHMR: Real-Time Human Mesh Recovery on Your Phone
- **类型：** repo
- **组织：** Naturalpad-studio（原 `mohamdev/InstantHMR` 已重定向）
- **代码：** <https://github.com/Naturalpad-studio/InstantHMR>
- **论文：** <https://arxiv.org/abs/2611.09195>（2026-11-06）
- **权重：** <https://huggingface.co/momolesang/InstantHMR>
- **训练：** <https://github.com/Naturalpad-studio/InstantHMR/blob/main/training/README.md>
- **训练数据：** <https://huggingface.co/datasets/facebook/sam-3d-body-dataset>
- **开源状态（步骤 2.5）：** 代码大部分 Apache-2.0；部分训练文件及权重为 SAM License，`nano/tiny/small/medium` 权重另限非商业，`medium-commercial` 在适用条款下可商用。
- **入库日期：** 2026-09-28；**本次复核：** 2026-11-11
- **通读：** 已完成
- **通读范围：** 官方 `README.md` 全文（包括 News、Models/Fit、Install/Demo、Python、TensorRT、Training、Licence）；`demo.py`、`instanthmr/pipeline.py`、`instanthmr/inference.py`、`instanthmr/models.py` 及 `training/README.md`，对照 arXiv 论文正文。

## 版本变动：为什么原来的 wiki 不准确

| 维度 | 2026-09-28 旧快照 | 2026-11-11 官方案例 |
|---|---|---|
| 仓库入口 | `mohamdev/InstantHMR` | 已迁至 `Naturalpad-studio/InstantHMR` |
| Backbone | RepViT-M1.5 | **HGNetV2 B0/B1/B2/B4** |
| crop | 224×224 | **288×288** |
| Decoder | 9-token | **71 queries = body 1 + 2D joints 70，4 层，8 heads** |
| 模型 | 单个 ONNX | 按模型/fit 导出的网络图 + **独立 `mhr-body.onnx`**（Python demo） |
| 安装 | `python install.py` | `pip install -r requirements.txt`（另装适配 PyTorch） |
| 新功能 | 无 | 2026-11-08 **kinematic fit/pose filters**；2026-11-10 **PyTorch checkpoints** |
| 许可 | Apache-2.0 泛化 | **代码/派生训练文件/权重按不同许可分别判断** |

## 源码核验与实际调用

1. `demo.py` 解析 `--model`（默认 medium）、`--fit`（默认 2）、`--device`、`--detector-stride` 等参数，构造 `PosePipeline`。
2. `instanthmr/pipeline.py`：`PersonDetector` 为 RF-DETR；每隔 N 帧检测，否则使用由上一帧 2D 关键点生成的框；返回 `FrameResult(persons, detector_ms, hmr_ms)`。
3. `instanthmr/models.py`：`resolve_model` 从 `momolesang/InstantHMR` 拉取对应 `instanthmr-{size}[-fit{1|2}].onnx`；`resolve_body` 获取 `mhr-body.onnx`。
4. `instanthmr/inference.py`：`InstantHMR.predict` 在 `--fit 0` 做 crop/归一化/CLIFF/ONNX 推理；`--fit 1/2` 走一个人一调用的多 crop 拟合图；可先对 MHR pose 作 `PoseFilter`，再调用 body 图获得 **70 个 3D 关键点 + 18,439 顶点**。
5. `training/README.md`：四档 backbone，基于 SAM 3D Body 发布 GT 拟合而非默认 teacher 蒸馏；训练入口 `training/train_ddp.py`，导出入口 `tools/export_onnx.py`，`--body` 另导出 body 图。

## 模型、速度和许可口径

论文表 3 的未启用 fit（`fit0`）数据：

| 模型 | EMDB-1 PA-MPJPE | RTX4070 TensorRT | Galaxy S23 NPU |
|---|---:|---:|---:|
| N | 61.7 mm | 0.70 ms | 4.60 ms |
| T | 58.0 mm | 0.73 ms | 4.84 ms |
| S | 53.3 mm | 0.84 ms | 5.38 ms |
| M | 48.8 mm | 0.98 ms | 5.49 ms |
| M-C | 72.2 mm | 0.98 ms | 5.49 ms |

**必须区分：** 论文基准测每个人体 crop 到最多 **6,890 顶点**的模型阶段；README 示例 demo 另做 detection/crop/body graph，端到端延迟不同。README 指出 medium 在 RTX 4070 的 detector 约 **27 ms**；不能据 **5.49 ms/人**推导“整机 180 FPS”。

**许可复核：** 上游 `LICENSE` 为 Apache-2.0；`LICENSE-SAM` 约束 `training/mhr_cont.py`、`training/assets/` 及发布权重/`mhr-body.onnx`；N/T/S/M 非商业权重另受训练图像用途限制，M-C 使用商业允许的 Harmony4D + COCO 子集，精度更低。应阅读最新原版许可证而不是仅依仓库 License badge。

## 对 wiki 的映射

- 主实体：**[InstantHMR](../../wiki/entities/instanthmr.md)**
- 同行论文：[论文通读记录](../papers/instant_hmr_arxiv_2611_09195.md)
- 上游：[SAM 3D Body](../../wiki/entities/sam-3d-body.md)
- C++ 对照：[SAM3DBody-cpp](../../wiki/entities/sam3dbody-cpp.md)
