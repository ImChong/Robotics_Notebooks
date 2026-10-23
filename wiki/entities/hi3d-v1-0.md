---
type: entity
project_id: hi3d-v1-0
project: https://www.hi3d.ai/hi3d-1-0-open-source
code: https://github.com/hi3d-official/Hi3Dv1.0
tags: [3d-generation, mesh, geometry-refinement, generative-ai, open-source, robotics-assets]
status: complete
updated: 2026-10-23
related:
  - ./3dgenstudio.md
  - ./meshroom.md
  - ../concepts/sim2real.md
sources:
  - ../../sources/sites/hi3d-v1-0-open-source.md
  - ../../sources/repos/hi3d-v1-0.md
summary: "Hi3D v1.0 是基于 Sparc3D 的高分辨率三维几何细化模型：以物体参考图和粗网格为条件逐级生成细节网格，适合视觉资产生产；推理代码和权重已公开，训练代码与评测工具未随仓库发布。"
---

# Hi3D v1.0

**Hi3D v1.0** 是基于 Sparc3D 架构的高分辨率 3D 几何细化模型：给定同一物体的参考图像和粗网格，模型逐级补充几何细节，输出更高分辨率的网格。核心仓库公开推理代码和权重；它不是单独从文本生成机器人结构或仿真配置的工具。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| 3D | Three-Dimensional | 本页讨论的三维物体几何表示 |
| VAE | Variational Autoencoder | 稀疏 VAE 将潜在表示解码为可提取的网格几何 |
| GPU | Graphics Processing Unit | 推理依赖 NVIDIA GPU 与特定 CUDA/PyTorch 环境 |
| CLI | Command-Line Interface | 仓库提供已有粗网格细化的命令行入口 |
| GLB | GL Transmission Format Binary | 示例中使用的三维网格交换格式之一 |

## 为什么重要

具身仿真和视觉研究常需要外观合理、局部结构清楚的三维道具。Hi3D 的价值在于把粗几何与参考图组合起来做几何细化，可作为资产生产链中的一个专用环节。它与 [3D Gen Studio](./3dgenstudio.md) 的工作台式生成流程互补：3D Gen Studio 编排多种模型和编辑步骤，Hi3D 则提供专门的网格细化模型。

进入机器人仿真还需要资产工程。与 [Meshroom](./meshroom.md) 的多视图摄影测量重建不同，Hi3D 通过参考图条件细化既有粗网格；与 [Sim2Real](../concepts/sim2real.md) 相关的尺度、碰撞、接触和动力学校验仍由下游流程负责。

## 核心原理

官方仓库将方法描述为基于 Sparc3D 表示和细化架构的逐级稀疏生成流程：

1. 将粗网格转换为稀疏坐标支撑，并编码参考图像特征。
2. 图像特征为粗几何位置上的稀疏流模型提供条件，逐轮采样稀疏潜变量。
3. 自剪枝稀疏 VAE 解码潜变量并提取网格。
4. 将新网格重新体素化，作为下一轮细化的支撑。

仓库公开的提取档位为 512³、1024³、1536³；这是稀疏坐标与提取网格的标称分辨率，不是密集潜变量张量大小，也不直接代表三角面数量。1536³ 流程较耗显存。

### 流程总览

```mermaid
flowchart LR
  A["参考图像 + 粗网格"] --> B["图像特征与稀疏坐标"]
  B --> C["逐级稀疏潜变量细化"]
  C --> D["自剪枝稀疏 VAE 解码"]
  D --> E["网格提取与重新体素化"]
  E --> F["高分辨率几何网格"]
```

## 工程实践

| 项目 | 官方仓库给出的信息 |
|------|------|
| **入口** | Python API `Sparc3DPipeline`；CLI `demo/refine_mesh.py` |
| **分辨率** | 512³、1024³、1536³ 三档；复杂输入可能使用更粗的稀疏支撑，达不到所请求档位时会报错 |
| **环境** | Linux、Python 3.10、NVIDIA GPU 计算能力 8.0+，CUDA toolkit 12.0–12.5 与 PyTorch 构建匹配 |
| **已验证配置** | README 报告 Python 3.10.12、PyTorch 2.5.0+cu124、CUDA 12.4 和 96 GB NVIDIA H20；1536³ 仅在该 GPU 上验证 |
| **在线试用** | 官方 README 链接到 `ilcve21/Sparc3D` Hugging Face Space；页面名称仍为 Sparc3D，需自行核对当前部署版本和运行状态 |

端到端示例可选接入 Hunyuan3D-2 mini 生成粗网格；Hi3D 核心推理也能直接接收用户已有的粗网格。不要把可选粗网格生成器误认为 Hi3D 的必需组件。

## 局限与风险

- **几何不是物理真值：** 参考图无法约束所有不可见表面；输出应视为条件补全，不代表物体真实背面或内部结构。
- **不能直接当机器人模型：** 网格不包含可信关节、质量、惯量、碰撞语义或 URDF/MJCF 配置，进入仿真前需要另行处理和验证。
- **复现范围有限：** 官方仓库提供推理实现，不含训练代码和论文评测工具；训练资产也未作为可下载数据集列出。
- **硬件门槛高：** 无 CPU 推理路径；1536³ 报告仅在 96 GB H20 上验证，不应推断较小显存卡也可运行。
- **许可有条件：** Hi3D Open License Agreement Version 1.0 允许商业使用，但对达到规定月活/收入规模的服务提出申请要求，并要求商业集成展示 Powered by Hi3D 1.0；使用前阅读协议全文及第三方许可。

## 关联页面

- [3D Gen Studio](./3dgenstudio.md) — 多模型 3D 资产生产与编辑工作台，可作为细化模型的上游/编排层。
- [Meshroom](./meshroom.md) — 从多视图照片进行摄影测量重建的离线路线。
- [Sim2Real](../concepts/sim2real.md) — 三维网格进入机器人仿真后的尺度、碰撞和物理一致性问题。

## 参考来源

- [Hi3D v1.0 官方项目页](../../sources/sites/hi3d-v1-0-open-source.md)
- [Hi3D v1.0 官方推理仓库](../../sources/repos/hi3d-v1-0.md)

## 推荐继续阅读

- [Hi3D v1.0 官方代码](https://github.com/hi3d-official/Hi3Dv1.0)
- [Hi3D v1.0 模型权重](https://huggingface.co/hi3d-official/Hi3Dv1.0)
- [Hugging Face 在线体验入口](https://huggingface.co/spaces/ilcve21/Sparc3D)
- [Sparc3D 论文](https://arxiv.org/abs/2505.14521)
