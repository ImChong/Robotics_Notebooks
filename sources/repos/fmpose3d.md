# FMPose3D（AdaptiveMotorControlLab/FMPose3D 官方仓库）

- **标题**: FMPose3D: monocular 3D pose estimation via flow matching
- **论文**: <https://arxiv.org/abs/2602.05755>（CVPR 2026）
- **项目页**: <https://xiu-cs.github.io/FMPose3D/>
- **代码**: <https://github.com/AdaptiveMotorControlLab/FMPose3D>
- **类型**: code-release（与 `sources/papers/fmpose3d_arxiv_2602_05755.md` 分工：本文件聚焦仓库、权重与 demo 入口）
- **机构**: 洛桑联邦理工学院（EPFL），Adaptive Motor Control Lab
- **License**: 代码 **Apache-2.0**；预训练模型 **CC BY-NC-ND（NC）**
- **首次入库**: 2026-07-17

## 一句话摘要

EPFL Adaptive Motor Control Lab 官方实现：用 **条件 Flow Matching** 做 **2D 关键点 → 3D 关节** 提升，支持人与动物；提供 **PyPI 包**、Hugging Face 自动权重下载、in-the-wild 单图 demo，并已 **集成进 DeepLabCut** 动物 3D 管线。

## 仓库时间线（README News，截至 2026-07-17）

- **2026-03** — 方法集成进 [DeepLabCut](https://www.mackenziemathislab.org/deeplabcut)。
- **2026-02** — CVPR 2026 接收；代码与 arXiv 发布。
- **PyPI** — `fmpose3d` 包可 `pip install`；动物管线可选 `fmpose3d[animals]`。

## 权重与许可

| 资产 | 入口 | 许可 |
|------|------|------|
| 人体 lifter | [Hugging Face MLAdaptiveIntelligence/FMPose3D](https://huggingface.co/MLAdaptiveIntelligence/FMPose3D) | NC（非商用） |
| 备用下载 | [Google Drive](https://drive.google.com/drive/folders/1aRZ6t_6IxSfM1nCTFOUXcYVaOk-5koGA) | 同 README 说明 |
| 动物 2D+3D | HF 自动下载（SuperAnimal-Quadruped + animal lifter） | 见 animals/README |

## 可复现入口（README 摘要）

- **环境：** Python **3.10**；`pip install fmpose3d`；Torch 钉 `>=2.4.1,<2.5`（CUDA 12.1 wheels）。
- **In-the-wild 人体 demo：** 图片放 `demo/images`，运行 `sh vis_in_the_wild.sh`；权重首次运行自动拉取。
- **训练：** Human3.6M 预处理 npz 可自 [VideoPose3D 流程](http://vision.imar.ro/human3.6m/) 或作者 [Drive 预处理包](https://drive.google.com/drive/folders/112GPdRC9IEcwcJRyrLJeYw9_YV4wLdKC)；`sh ./scripts/FMPose3D_train.sh`。
- **评测：** `sh ./scripts/FMPose3D_test.sh`。
- **Inference API：** `fmpose3d/inference_api/README.md` 提供端到端图像→3D API。
- **动物：** 见 `animals/README.md`；demo 自动下载 26-joint 2D 检测器 + 3D lifter。

## 与机器人 / 行为分析栈的关系

- **上游感知：** 输出 **3D 关键点序列**（非 SMPL 网格），可作为动物行为分析、遥操作视觉追踪或后续重定向前的 **稀疏 3D 姿态源**。
- **DeepLabCut 生态：** 与实验室既有 **markerless 动物姿态** 工具链直接对接，降低「2D DLC → 3D」工程门槛。
- **对比扩散 lifter：** 论文报告相对 DiffPose **~5.4×** 推理加速（$N=40$ 假设仍 **>145 FPS** on RTX 4090），适合需要 **在线多假设** 的场景。

## 对 Wiki 的映射

- **`wiki/entities/paper-fmpose3d-monocular-3d-pose-flow-matching.md`**：论文实体与方法归纳。
- **`sources/papers/fmpose3d_arxiv_2602_05755.md`**：论文级摘录。
- **`wiki/concepts/motion-retargeting-pipeline.md`**：动物/关键点上游源交叉引用。
