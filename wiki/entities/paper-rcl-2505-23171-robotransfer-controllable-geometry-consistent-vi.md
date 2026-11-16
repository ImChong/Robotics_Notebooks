---
type: entity
project_id: horizon-robotransfer
tags: [paper, curated-index, awesome-world-action-models-rcl, rcl-wam-catalog, repo, open-source, china-embodied-opensource, project]
status: complete
updated: 2026-11-16
arxiv: "2505.23171"
code: https://github.com/HorizonRobotics/RoboTransfer
venue: "arXiv preprint"
institutions:
  - horizon-robotics
summary: "RoboTransfer（arXiv:2505.23171，地平线 HorizonRobotics，已开源 HorizonRobotics/RoboTransfer）：在保留真实演示多视角几何（metric depth + 法线）的同时可控替换视觉外观，实现多视角合成与 ACT 策略迁移；含源码运行时序图。"
related:
  - paper-rcl-wam-robot-learning-control-survey.md
  - ../overview/rcl-awesome-wam-technology-map.md
  - ../methods/generative-world-models.md
  - ../methods/vla.md
  - ../tasks/manipulation.md
  - ../tasks/locomotion.md
  - ../concepts/world-action-models.md
  - ../overview/china-domestic-embodied-opensource-76-companies-technology-map.md
  - ../entities/humanoid-motion-intelligence.md
  - ../queries/china-domestic-opensource-424-coverage.md
  - ../queries/hmi-opensource-projects-coverage.md
sources:
  - ../../sources/papers/rcl_awesome_wam_2505_23171_robotransfer-controllable-geometry-consi.md
  - ../../sources/papers/rcl_awesome_wam_catalog.md
  - ../../sources/repos/awesome-world-action-models-rcl.md
  - ../../sources/repos/robotransfer.md
  - ../../sources/blogs/wechat_embodied_station_domestic_opensource_panorama_2026-09-06.md
---

# RoboTransfer

**RoboTransfer: Controllable Geometry-Consistent Video Diffusion for Manipulation Policy Transfer**（[arXiv:2505.23171](https://arxiv.org/abs/2505.23171)，[GitHub](https://github.com/HorizonRobotics/RoboTransfer)）由**地平线（Horizon Robotics）**公开，既收录于 [Awesome World-Action Models (RCL)](https://github.com/rcl-robotics/Awesome-World-Action-Models) **第 119/564** 篇（分组 **Datasets**），也作为 [国内具身智能开源全景（76 家 · 424 项）](../overview/china-domestic-embodied-opensource-76-companies-technology-map.md) 第四层「世界模型」代表开源项目归档。本页合并论文机制拆解与官方源码时序。

## 一句话定义

**RoboTransfer 以真实演示的度量深度（metric depth）与表面法线维持几何一致性，用可替换的背景修补图与物体语义条件驱动多视角视频扩散生成，以合成轨迹与真实数据混合提升操作策略迁移。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| WAM | World Action Model | 世界预测与动作生成耦合 |
| VLA | Vision-Language-Action | 视觉–语言–动作策略 |
| IDM | Inverse Dynamics Model | 先预测未来再反推动作 |
| WM | World Model | 环境前向预测模型 |
| VAE | Variational Autoencoder | 联合编码多视角图像与几何引导图的潜空间编解码器 |

## 为什么重要

- RoboTransfer augments robot demonstrations by changing their visual appearance while conditioning video diffusion on the demonstrated geometry. Jointly encoded camera views, metric depth, normals and separate background/object references support multi-view synthesis. A separately trained ACT policy benefits from the generated data.
- 想横向对照同一分组的其他工作，可以从 [RCL Awesome WAM 技术地图](../overview/rcl-awesome-wam-technology-map.md) 逐条展开，并通过 [国内具身开源全景技术地图](../overview/china-domestic-embodied-opensource-76-companies-technology-map.md) 查看地平线同生态其它仿真/世界模型项目。
- 顺着列表实体 [Awesome World-Action Models](paper-rcl-wam-robot-learning-control-survey.md)、[Humanoid Motion Intelligence](../entities/humanoid-motion-intelligence.md) 与站内 WAM / VLA 方法页，可以接回对应的学习主线。

## 流程总览

```mermaid
flowchart TD
    subgraph DATA["数据构建 §3.3: 真实演示拆成三元组"]
        R["真实机器人演示<br/>头部相机 + 双腕相机"] --> G1["几何条件<br/>深度估计器 + RGB-D 尺度对齐(鲁棒最小二乘)得 metric depth<br/>单目法线估计得 surface normal"]
        R --> K["多视角 RGB 关键帧"]
        K --> V["VLM 描述器生成场景/物体描述<br/>Grounding-SAM 分割物体"]
        V --> BG["物体修补(inpaint)得无物体背景参考图 C_b"]
        V --> OB["物体 mask 经 CLIP 得语义嵌入 C_o"]
    end
    subgraph GEN["RoboTransfer 生成 §3.2: 多视角视频扩散"]
        MV["N 路同步视频沿宽度拼接<br/>VAE 联合编码, 无需改动扩散骨干"]
        G1 --> GE["几何 VAE(卷积)编码 depth + normal 视频<br/>与噪声潜变量按通道拼接"]
        BG --> BE["VAE 编码背景参考图<br/>与潜变量拼接, 控制全局纹理"]
        OB --> OE["CLIP 全局嵌入经 cross attention<br/>作为非结构化物体外观条件"]
        MV --> D["视频扩散去噪"]
        GE --> D
        BE --> D
        OE --> D
    end
    D --> S["几何一致、外观可控的多视角合成视频"]
    S --> P["与真实数据混合训练 ACT 策略<br/>论文实验 50/50 混合最佳(§5.2)"]
```

图依据：论文 §3.2 RoboTransfer Framework、§3.3 Dataset Construction / Fig.2–3、§5 Real Robot Experiments。

## 核心信息

| 字段 | 内容 |
|------|------|
| 机构 | 地平线（Horizon Robotics） |
| 编号 / 分组 | RCL WAM #119/564（Datasets · 合成数据与数据生成）；国内具身开源全景第四层（世界模型） |
| 出处 | arXiv preprint |
| 论文 | <https://arxiv.org/abs/2505.23171> |
| 代码 | <https://github.com/HorizonRobotics/RoboTransfer> |
| 官方组织 | <https://github.com/HorizonRobotics> |

## 核心机制（归纳）

### 策展导读要点

RoboTransfer augments robot demonstrations by changing their visual appearance while conditioning video diffusion on the demonstrated geometry. Jointly encoded camera views, metric depth, normals and separate background/object references support multi-view synthesis. A separately trained ACT policy benefits from the generated data.

若需工程落地，请结合下方源码运行时序图，并对照站内 [World Action Models（WAM）](../concepts/world-action-models.md) 等概念页与官方仓库 README。

## 源码运行时序图

入口：`uv run main.py`（显存较小的 4090 加 `--mem_efficient`；默认 `--dataset_path HorizonRobotics/RoboTransfer-RealData`、`--refer_image_path assets/example_ref_image/gray_grid_desk.png`）

```mermaid
sequenceDiagram
    participant U as 用户 uv run main.py
    participant M as main.py main()
    participant D as datasets.load_dataset / utils/image_loading.py
    participant P as robotransfer_pipeline.RoboTransferPipeline
    participant GN as guider.GuiderNet（depth / normal）
    participant UN as UNetSpatioTemporalConditionModel
    participant V as vae（AutoencoderKLTemporalDecoder）
    participant W as utils/save_video.save_images_to_mp4

    U->>M: 解析 --dataset_path / --refer_image_path / --output_dir / --mem_efficient
    M->>D: HF 数据集用 load_dataset(split="train")，本地路径用 get_dataset_length
    M->>P: from_pretrained("HorizonRobotics/RoboTransfer", bfloat16, trust_remote_code=True)
    M->>P: pipe.to("cuda")
    opt --mem_efficient
        M->>P: enable_model_cpu_offload()
    end
    loop 每 30 帧一段：range(0, length-30, 30)
        M->>D: load_images_from_dataset / load_images_from_local（取 depth、normal 引导图）
        D-->>M: depth_guider_images, normal_guider_images
        M->>P: pipe(image=参考图, 引导图, 30 帧, 25 步, 高 384, 宽 640x3)
        P->>P: _encode_image（CLIP 图像编码）与 _encode_vae_image（参考图潜变量）
        P->>GN: depth_guider_net / normal_guider_net 编码逐帧引导图
        P->>P: prepare_noise_latents，guidance_scale 在 min..max 间 linspace
        loop 每个去噪时间步
            P->>UN: unet(拼接 latents、图像潜变量、depth 条件、normal 条件, image_embeddings)
            UN-->>P: noise_pred（带 classifier-free guidance）
            P->>P: scheduler.step
        end
        P->>V: decode_latents 分块解码（decode_chunk_size=1）
        P-->>M: frames[0]
        M->>W: save_images_to_mp4(frames, output_dir/output_frames.mp4, fps=10)
    end
```

图依据：HorizonRobotics/RoboTransfer@3698fee，读过 `README.md`、`main.py`、`robotransfer/models/robotransfer_pipeline.py`（`__init__` 与 `__call__`，去噪循环后段未逐行读）、`robotransfer/models/guider.py`（仅类与方法签名）、`robotransfer/utils/` 下文件名与函数签名。

## 工程实践

- **官方仓库运行**：从 [HorizonRobotics/RoboTransfer](https://github.com/HorizonRobotics/RoboTransfer) 核对 README 许可、Hugging Face 模型权重要求与显存选项（如 24GB 显卡使用 `--mem_efficient` 开启 `enable_model_cpu_offload`）。
- **同生态对照**：通过 [424 项覆盖索引](../queries/china-domestic-opensource-424-coverage.md) 对照地平线同组织下的仿真器与世界模型工具链。

## 评测与指标

- 本页 **没有搬运** 原文的量化 benchmark 与实机指标。
- 评测口径与具体数值以 [原文 PDF](https://arxiv.org/abs/2505.23171) 为准（论文 §5.2 关注合成视频与真实数据 50/50 混合训练 ACT 策略的真机迁移增益）。
- 横向对照请回到 [技术地图](../overview/rcl-awesome-wam-technology-map.md) 同分组条目。

## 与其他工作对比

- 本页 **不做** 与具体基线的逐项数值对比；同分组的横向对照请回到 [技术地图](../overview/rcl-awesome-wam-technology-map.md) 的 **Datasets** 分组逐条展开。
- 本页已合并论文框架拆解与 [HorizonRobotics/RoboTransfer](https://github.com/HorizonRobotics/RoboTransfer) 推理源码时序分析；训练脚本与数据集构建细节仍以原文和官方仓为准。

## 结论

**RoboTransfer 定位为面向操作策略数据增强的几何一致多视角视频扩散模型：核心是用深度与法线锁定几何结构、只泛化背景与物体外观，并已通过 `HorizonRobotics/RoboTransfer` 开放推理代码与权重。**

- 关键设计在于把真实演示拆解成「几何条件（metric depth + normal）+ 无物体背景 + 物体语义条件」，将跨视图视频沿宽度拼接联合编码，使生成数据可直接用于 ACT 策略混合训练。
- 工程可复现性较明确：官方仓库 `HorizonRobotics/RoboTransfer` 提供 `uv run main.py` 推理流水线（含 `GuiderNet` 与 `RoboTransferPipeline`），具体许可与完整训练管线以仓库 README 为准。
- 深读与定量对比仍需结合论文原文 PDF 验证。

## 常见误区

1. 不要把 Awesome 条目的 Contribution 当成完整方法证明——量化对比与消融须回看原文 PDF。
2. 推理源码时序图仅覆盖前向视频合成（`main.py`），不包括下游 ACT 策略混合训练或原始深度/法线数据构建流水线。

## 关联页面

- 列表实体：[Awesome World-Action Models（RCL）](paper-rcl-wam-robot-learning-control-survey.md)
- 技术地图：[RCL Awesome WAM 技术地图](../overview/rcl-awesome-wam-technology-map.md)
- [World Action Models（WAM）](../concepts/world-action-models.md)
- 方法/任务：[generative-world-models.md](../methods/generative-world-models.md)、[manipulation.md](../tasks/manipulation.md)
- [国内具身开源全景技术地图](../overview/china-domestic-embodied-opensource-76-companies-technology-map.md)
- [HMI 开源项目主表导读](../queries/hmi-opensource-projects-coverage.md)
- [Humanoid Motion Intelligence](../entities/humanoid-motion-intelligence.md)

## 参考来源

- [`sources/papers/rcl_awesome_wam_2505_23171_robotransfer-controllable-geometry-consi.md`](../../sources/papers/rcl_awesome_wam_2505_23171_robotransfer-controllable-geometry-consi.md) — 本条目策展摘录
- [`sources/papers/rcl_awesome_wam_catalog.md`](../../sources/papers/rcl_awesome_wam_catalog.md) — 列表总表
- [`sources/repos/awesome-world-action-models-rcl.md`](../../sources/repos/awesome-world-action-models-rcl.md)
- [RoboTransfer 源码归档](../../sources/repos/robotransfer.md)（<https://github.com/HorizonRobotics/RoboTransfer>）
- [国内具身智能开源全景（微信公众号）](../../sources/blogs/wechat_embodied_station_domestic_opensource_panorama_2026-09-06.md)
- [`docs/PAPERS.md`](https://github.com/RCL-Robotics/Awesome-World-Action-Models/blob/main/docs/PAPERS.md) — 上游论文目录
- 论文：<https://arxiv.org/abs/2505.23171>
- [arXiv PDF](https://arxiv.org/pdf/2505.23171)：已通读方法部分（实验与附录以略读为准）：§3.2 框架、§3.3 数据构建、§5 真机实验设定与合成数据比例（略读）
- 源码核对（2026-11-02）：HorizonRobotics/RoboTransfer@3698fee，读过 `README.md`、`main.py`、`robotransfer/models/robotransfer_pipeline.py`（`__init__` 与 `__call__` 大部分）、`guider.py` 与 `utils/` 的函数签名；时序图依据以上文件，未运行代码。

## 推荐继续阅读

- [RoboTransfer 官方 GitHub 仓库](https://github.com/HorizonRobotics/RoboTransfer)
- [地平线 官方组织](https://github.com/HorizonRobotics)
- [Awesome World-Action Models (RCL) 仓库](https://github.com/rcl-robotics/Awesome-World-Action-Models)
- [原文](https://arxiv.org/abs/2505.23171)
