---
type: entity
project_id: instanthmr
code: https://github.com/Naturalpad-studio/InstantHMR
arxiv: "2611.09195"
tags: [perception, human-mesh-recovery, mhr, onnx, deployment, real-time, open-source, single-image]
status: complete
updated: 2026-11-11
related:
  - ../queries/robot-perception-stack-selection-loop.md
  - ./sam-3d-body.md
  - ./sam3dbody-cpp.md
  - ../concepts/motion-retargeting-pipeline.md
  - ../concepts/whole-body-tracking-pipeline.md
  - ./rf-detr.md
  - ../methods/motion-retargeting-gmr.md
sources:
  - ../../sources/papers/instant_hmr_arxiv_2611_09195.md
  - ../../sources/repos/instanthmr.md
  - ../../sources/repos/sam-3d-body.md
original:
  - ../../sources/papers/instant_hmr_arxiv_2611_09195.md
  - ../../sources/repos/instanthmr.md
summary: "InstantHMR 是基于 SAM 3D Body 已发布 MHR 标注训练的手机端实时单目人体网格恢复模型族：HGNetV2 + 71-query Transformer，288×288 人体裁剪，M 型在 Galaxy S23 NPU 上以 5.49 ms/人重建 6,890 顶点网格；开源 ONNX、运动学拟合与训练代码。"
---

# InstantHMR｜Real-Time Human Mesh Recovery on Your Phone（手机端实时人体网格恢复）

**InstantHMR**（[论文 arXiv:2611.09195](https://arxiv.org/abs/2611.09195)；[Naturalpad-studio 官方源码](https://github.com/Naturalpad-studio/InstantHMR)）由 Mohamed Adjel、Kahina Chalabi、Vincent Bonnet 提出，基于 Meta **Momentum Human Rig（MHR）** 参数化人体模型，兼顾单目 3D 人体网格恢复的**手机 NPU 延迟**与精度。旧仓库 `mohamdev/InstantHMR` 已重定向至 `Naturalpad-studio/InstantHMR`；下文依照 **2026-11-06 论文、2026-11-10 代码/权重更新**，不再沿用早期 RepViT / 224 像素 / 9-token 描述。

## 一句话定义

**单张 RGB + 检测框 → 288×288 人体裁剪 → HGNetV2 + 71-query Transformer → MHR 参数、2D 关键点与相机平移 → MHR 骨架/稠密网格**。论文中的 **InstantHMR-M** 在 **Galaxy S23 NPU** 上重建 **6,890 顶点**网格耗时 **5.49 ms/人**，在 **RTX 4070 TensorRT** 上为 **0.98 ms/人**；两者均**不包含检测和裁剪**（论文 `4/表 1）。

## 英文缩写速查

| 缩写 | 英文全称 | 作用 |
|---|---|---|
| HMR | Human Mesh Recovery | 单图估计人体的 3D 姿态、形状与网格 |
| MHR | Momentum Human Rig | Meta 的参数化人体骨架与表面模型 |
| NPU | Neural Processing Unit | 手机侧神经网络加速器 |
| CLIFF | Carrying Location Information in Full Frames | 将完整图像内的框位置与相机焦距纳入条件 |
| SimCC | SimCC coordinate classification | 分别预测关键点 x/y 的离散位置分布 |
| PA-MPJPE | Procrustes-Aligned Mean Per Joint Position Error | 刚性对齐后的平均关节位置误差，越小越好 |
| ONNX | Open Neural Network Exchange | 模型交换与多后端推理格式 |

## 为什么重要

- **边缘实时推理**：论文比较 11 个已有方法，统一将每个人体裁剪推理到最多 6,890 顶点的姿态网格；M 型在手机 NPU 上比精度更高的 NLF-S / NLF-L 分别快 **12× / 23×**。这是**论文指定测试协议和硬件**下的结论，不等同于整机端到端帧率。
- **机器人动捕输入**：获得 70 个 2D/3D 关键点、MHR 姿态及相机系网格，可作为 [运动重定向](../concepts/motion-retargeting-pipeline.md) 或 [全身跟踪](../concepts/whole-body-tracking-pipeline.md) 的上游；但**不是**能直接驱动机器人关节的策略。
- **可部署、可复现**：官方公开模型族、ONNX 图、训练/数据流水线、TensorRT/ONNX Runtime demo；最新版提供 **`--fit` 运动学拟合**和视频滤波。
- **许可证必须分层**：代码大部分为 Apache-2.0，但部分训练文件和权重受 SAM License 约束；常用 N/T/S/M 权重**仅限非商业用途**，`medium-commercial` 可按相关条款商用但精度下降。不能用“仓库 Apache-2.0”推导全部权重均可商用。

## 核心原理

### 输入、网络与输出

| 环节 | 来自论文/官方源码的事实 |
|---|---|
| 输入 | 逐人 **288×288 RGB crop**；3 维 CLIFF 条件包含检测框相对图像中心的视角与框宽/焦距 |
| Backbone | **HGNetV2 B0/B1/B2/B4**（分别对应 nano/tiny/small/medium），得到 **9×9 = 81** 个图像 token，512 维 |
| Decoder | **4 层、8 heads** Transformer；**71 个 learned queries**（1 个全身 + 70 个 2D 关键点） |
| 人体分支 | **204 维 MHR rig 参数**（姿态、骨长等）+ **45 维 shape 参数** |
| 关键点分支 | SimCC 为 **70 个 2D 关键点**分别输出 x/y **96-bin** 分布；**3D 关键点由 MHR 正向运动学得到**，不是另起一个 3D 回归头 |
| 相机分支 | 3 维相机平移；无标定时官方 demo 默认焦距为画面对角线的 **1.05 倍** |
| 网格 | 论文测速用 MHR **6,890 顶点**子集以与 SMPL 方法公平比较；官方 Python demo 的 `mhr-body.onnx` 可提供 **18,439 顶点**完整网格 |

训练使用 Meta SAM 3D Body 公开的**人工标注 MHR 拟合**（**非**用教师预测伪标签做默认蒸馏）：SA-1B、AI Challenger、Harmony4D、COCO 和 MPII 共约 **420 万个人体裁剪**。11 项损失覆盖参数、骨架/顶点几何及 2D/reprojection 约束；标准 crop 75% + 其他三种 crop 25%，论文消融表明比只用标准 crop 更准。不要把该 2026 年论文方法误记作旧 README 快照中的 RepViT-M1.5 + 9-token。

### 推理流程图（论文图 2 + 源码）

~~~mermaid
flowchart LR
  A["RGB 图像 / 视频帧"] --> B["RF-DETR 检测或复用跟踪框"]
  B --> C["288×288 crop + CLIFF 相机条件"]
  C --> D["HGNetV2: 9×9 图像 token"]
  D --> E["4 层 Transformer: 1 body + 70 keypoint queries"]
  E --> F["MHR 参数 204 + shape 45 + 相机平移 3"]
  E --> G["SimCC: 70 个 2D 关键点"]
  F --> H{"--fit 1/2?"}
  G --> H
  H -->|是| I["LM 运动学拟合：对齐 2D 关键点"]
  H -->|否| J["直接使用预测的 MHR 参数"]
  I --> K["mhr-body.onnx 正向运动学 + 蒙皮"]
  J --> K
  K --> L["70 个 3D 关键点 + 人体网格"]
  L --> M["相机坐标系输出 / Rerun / 重定向前处理"]
~~~

**版本差异**：论文 v1 的主体模型基准是网络直接推理（`--fit 0`）；官方仓库 **2026-11-08 v1.1** 新增网络图内的 kinematic fit，`--fit 2` 成为 demo 默认。拟合采用 1 步 Levenberg–Marquardt，利用网络自己输出的 2D 关键点优化 MHR 姿态；它**不是额外联合训练的神经网络**，也不能把 fit 后精度和论文 fit0 延迟混用。

### 源码时序图（`demo.py` / `PosePipeline` / `InstantHMR.predict`）

~~~mermaid
sequenceDiagram
  autonumber
  participant U as demo.py
  participant P as PosePipeline
  participant D as PersonDetector / RF-DETR
  participant I as InstantHMR / ORT
  participant B as MHR body ONNX
  participant V as RerunVisualizer
  U->>P: RGB frame, timestamp
  alt detector_stride 到检测帧
    P->>D: image_rgb
    D-->>P: bbox + confidence
  else 非检测帧
    P->>P: 使用上一帧关键点更新的 bbox
  end
  P->>I: predict(image_rgb, bboxes, timestamp)
  I->>I: crop, CLIFF, ONNX 网络推理
  opt fit 1 / fit 2
    I->>I: 一/两个 crop + LM kinematic fit
  end
  opt 视频 pose-filter
    I->>I: one-euro 或 alpha-beta 平滑 MHR 参数
  end
  I->>B: mhr_params, shape_params
  B-->>I: 70 joints_3d + 18439 vertices
  I-->>P: HMRPrediction[]
  P->>P: keypoint_box 更新下一帧检测框
  P-->>U: FrameResult + 各阶段耗时
  U->>V: log(frame, persons, timings)
~~~

**源码边界**：`instanthmr/inference.py` 会再调用**独立的 `mhr-body.onnx`** 输出 18,439 顶点；“单个 ONNX 文件完成全流程、无需 body graph”不符合当前 Python demo。论文中的计时图则把 6,890 顶点 MHR FK/蒙皮打包计算，评测口径不同。

## 定量结果与选型

| 模型（论文 fit0） | 参数量 | EMDB-1 PA-MPJPE ↓ | 3DPW PA-MPJPE ↓ | RTX 4070 TRT | S23 NPU |
|---|---:|---:|---:|---:|---:|
| nano / HGNetV2-B0 | 20.0M | 61.7 mm | 46.6 mm | 0.70 ms | 4.60 ms |
| tiny / B1 | 20.3M | 58.0 mm | 44.0 mm | 0.73 ms | 4.84 ms |
| small / B2 | 24.4M | 53.3 mm | 41.7 mm | 0.84 ms | 5.38 ms |
| medium / B4 | 32.2M | **48.8 mm** | **40.0 mm** | **0.98 ms** | **5.49 ms** |
| medium-commercial / B4 | 32.2M | 72.2 mm | 50.7 mm | 0.98 ms | 5.49 ms |

数据来自[论文表 3](https://arxiv.org/abs/2611.09195)，**仅模型单人裁剪推理、含最多 6,890 顶点网格，不含 RF-DETR、预处理、视频 I/O**。M 型在 RICH 的 PA-MPJPE 为 **41.7 mm**。论文统一基准使用指定的真值框/标定焦距和 Meta 官方 MHR→SMPL 转换，不能与他人的不同评估协议数字直接排名。

## 工程实践

1. **安装**：`git clone https://github.com/Naturalpad-studio/InstantHMR.git`；Python **3.10–3.12**，按[官方 README](https://github.com/Naturalpad-studio/InstantHMR#install)安装 PyTorch 和 `requirements.txt`。
2. **单图**：`python demo.py --image photo.jpg`；权重首次使用从 [Hugging Face](https://huggingface.co/momolesang/InstantHMR) 下载，默认模型为 `medium`、`--fit 2`。
3. **视频/相机**：`python demo.py --video clip.mp4 --pose-filter one-euro` 或 `python demo.py --camera 0 --model small --fit 1`；可用 `--detector nano`、`--detector-stride 3` 缓解约 **27 ms** 的 RF-DETR medium 检测瓶颈（官方 RTX 4070 示例，非手机全链路测速）。
4. **机器人接口**：`PosePipeline(image).persons` 提供 `joints_3d_cam (70,3)`、`vertices_cam (18439,3)`、`mhr_params (204,)` 和 `shape_params (45,)`。相机系为 x 右、y 下、z 前，长度单位为米；重定向前需做坐标变换、尺度/接触校验以及 MHR→目标骨架映射。
5. **训练/导出**：`training/train_ddp.py` 使用 SAM 3D Body 已发布拟合与 4 GPU 训练配方；`tools/export_onnx.py` 导出网络/fit 图，`--body` 另导出共享 `mhr-body.onnx`。2026-11-10 起提供五种模型的 PyTorch checkpoints 便于 fine-tune（检查各自权重许可）。

**拟合选择**：以 M 型为例，官方表中 `fit 0/1/2` 的 EMDB-1 误差是 **48.8 / 45.2 / 44.4 mm**，S23 NPU 延迟分别是 **5.49 / 6.40 / 10.22 ms**；提高精度要付出额外延迟，选型时单独记录 `fit`。

## 局限与风险

- **手部细节**：没有独立手部 crop/解码器；288×288 全身图中手只占少量像素，不适合直接替代高精度手指动捕。
- **跨帧稳定性**：模型逐帧预测；论文**没有**报告时间抖动/加速度误差。仓库提供 One Euro / alpha-beta 过滤，但不能把单帧 PA-MPJPE 当成连续动作质量保证。
- **硬件外推**：论文手机测试针对 Galaxy S23/S24 的 Qualcomm NPU；虽然源码列出 CoreML 执行后端，**论文未测试 Apple Neural Engine**。
- **检测与相机误差**：论文速度不含检测与裁剪；未知相机内参时以焦距回退近似，深度、全局轨迹及重定向姿态可能偏差。
- **授权边界**：Apache-2.0 仅覆盖大部分代码；`training/mhr_cont.py`、`training/assets/` 和权重受 SAM License 约束，非商业版的训练图像也额外限制用途。商用前逐项审查[上游许可证](https://github.com/Naturalpad-studio/InstantHMR#licence)。

## 关联页面

- [SAM 3D Body](./sam-3d-body.md)：MHR 及官方基础模型
- [SAM3DBody-cpp](./sam3dbody-cpp.md)：另一条 C++ 运行与 BVH 路线
- [Motion Retargeting Pipeline](../concepts/motion-retargeting-pipeline.md)：将 MHR 姿态映射到机器人骨架
- [Whole-Body Tracking Pipeline](../concepts/whole-body-tracking-pipeline.md)：动捕感知数据链
- [RF-DETR](./rf-detr.md)：demo 的检测阶段

## 推荐继续阅读

- [InstantHMR 正式论文（arXiv:2611.09195）](https://arxiv.org/abs/2611.09195)
- [官方代码与 README](https://github.com/Naturalpad-studio/InstantHMR)
- [官方训练与微调指南](https://github.com/Naturalpad-studio/InstantHMR/blob/main/training/README.md)

## 参考来源

- [论文原文通读归档](../../sources/papers/instant_hmr_arxiv_2611_09195.md)
- [官方源码与 README 通读归档](../../sources/repos/instanthmr.md)
- [SAM 3D Body 来源](../../sources/repos/sam-3d-body.md)
