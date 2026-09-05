# NVIDIA Omniverse NuRec 文档（26.04）

> 来源归档（ingest）

- **标题：** NVIDIA Omniverse NuRec
- **类型：** site（产品文档）
- **发布方：** NVIDIA
- **原始链接：** <https://docs.nvidia.com/nurec/>
- **版本核查：** 26.04（2026-09-05）
- **配套论文 / 仓：** Instant NuRec arXiv:2607.14203；<https://github.com/NVIDIA/instant-nurec>
- **项目页：** <https://research.nvidia.com/labs/sil/projects/instant-nurec/>
- **入库日期：** 2026-09-05
- **一句话说明：** 把真实相机 / LiDAR 吃成可仿真 3D 环境的 **模型 + 服务**：重建写 **USDZ**，渲染走 **gRPC / Omniverse Kit**（`OmniNuRecVolumeAPI`），可选 Harmonizer 调和时序与外观；机器人走 Isaac Sim，自动驾驶走 NCore + 可选 Instant 初始化。

## 文档要点

### 重建产物

USDZ 内含：USD 场景、AI 重建 checkpoint（高斯位置与辅助量）、序列 track / 车体轨迹 JSON；可选 **OpenDRIVE (XODR)** 供交通建模。不是「只有一张 splat 图」。

### 渲染与精修

- **gRPC API** 连接仿真运行时与 NuRec 容器。
- Omniverse Kit 应用（含 **Isaac Sim**）加载兼容场景后，用 **`OmniNuRecVolumeAPI`** 控渲染。
- **Harmonizer** 提高时序一致性与外观调和。多 GPU 训练时 **不能** 在训练期开 Harmonizer，可放到推理 gRPC。

### 两条开发者入口（文档 Quickstart）

| 受众 | 路径 |
|------|------|
| **机器人** | 单目或双目重建 → USDZ → Isaac Sim 神经体积渲染。立体参考栈：Isaac ROS + cuSFM + FoundationStereo + nvblox + **3DGRUT** 导出 USDZ。也可直接用 Hugging Face **PhysicalAI Robotics NuRec** 预重建场景。高斯 **无固有碰撞**，Isaac 里常需另加 Ground Plane / 对齐 mesh。 |
| **自动驾驶** | 数据转 **NCore** + `nre-tools` 辅助量 → 容器 `nvcr.io/nvidia/nre/nre-ga:26.04` `mode=train`。文档 **推荐 Instant NuRec** 先出合并 PLY，再 `model/gaussians/initialization@...=nrm_ply` 做逐场景精修。预重建 AV 场景见 Physical AI NCore 数据集。 |

### Instant NuRec 在文档中的角色

[Reconstruct an AV Scene](https://docs.nvidia.com/nurec/nurec/reconstruct-av-scene.html) 把 Instant 写成 **推荐训练路径**：`./setup.sh` → `run_inference.py --merge` → 把 `merged.ply` 挂进 Docker 训练（默认初始化点数 2M，与 `--n-gaussians` 对齐）。无 Instant 种子的 Standard Training 仍可跑，迭代更多。

### 容器与门禁

- 拉镜像要 **NGC API key**。
- 与 Instant 仓分工：Instant 是 **本机 Python、秒级预览**；NuRec 是 **Docker 逐场景精修 → 高保真 USDZ**。输入同为 NCore V4 / HF / `sequence.json`。

## 论文 / 代码状态（步骤 2.5，2026-09-05）

- **Instant 推理仓：** 已开源（部分，Apache-2.0）— [nvidia-instant-nurec.md](../repos/nvidia-instant-nurec.md)
- **NuRec 训练 / 渲染容器：** NGC 产品镜像，**不是** 本仓库式开源；文档与预重建数据集公开获取
- **机器人 3DGRUT 路径：** 文档指向开源 3DGRUT 训练 + `export_usdz`；与 Instant 的 AV 前向路径不同
- **Isaac 消费：** Flexion × Niantic 联合文把现场 3DGS+mesh 打成 **NuRec volume USDZ** 进 Isaac Lab（见 [flexion 博客归档](../blogs/flexion_niantic_nvidia_sim2real_rgb_2026-07-20.md)）

## 对 wiki 的映射

- [NVIDIA Omniverse NuRec](../../wiki/entities/nvidia-nurec.md)
- [Instant NuRec 论文实体](../../wiki/entities/paper-instant-nurec.md)
- [Isaac Gym / Isaac Lab](../../wiki/entities/isaac-gym-isaac-lab.md)
- [Isaac Sim](../../wiki/entities/isaac-sim.md)
- [Flexion × Niantic × NVIDIA RGB 管线](../../wiki/entities/flexion-niantic-nvidia-rgb-sim2real-pipeline.md)
