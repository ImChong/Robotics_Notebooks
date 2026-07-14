# Terrain Diffusion（InfiniteDiffusion 开源实现）

> 来源归档

- **标题：** Terrain Diffusion / InfiniteDiffusion
- **类型：** repo / procedural-terrain / diffusion / world-generation / minecraft-mod
- **作者：** Alexander Goslin（Independent Researcher）
- **项目页：** <https://xandergos.github.io/terrain-diffusion/>
- **论文：** SIGGRAPH 2026 · [arXiv:2512.08309](https://arxiv.org/abs/2512.08309) · [DOI:10.1145/3799902.3811080](https://doi.org/10.1145/3799902.3811080)
- **代码：** <https://github.com/xandergos/terrain-diffusion>
- **相关仓库：**
  - [infinite-tensor](https://github.com/xandergos/infinite-tensor) — 无界张量框架
  - [terrain-diffusion-mc](https://github.com/xandergos/terrain-diffusion-mc) — Minecraft Fabric mod
- **Modrinth：** <https://modrinth.com/mod/terrain-diffusion>
- **模型：** [Hugging Face Collection](https://huggingface.co/collections/xandergos/terrain-diffusion)
- **许可：** MIT（Modrinth 标注）
- **入库日期：** 2026-07-14
- **一句话说明：** SIGGRAPH 2026 **InfiniteDiffusion** 的官方 Python 实现与 **Terrain Diffusion** 分层地形模型；提供探索 GUI、REST 式 API、Azgaar→GeoTIFF 条件导出，以及可游玩的 **Minecraft Fabric** 世界类型集成。
- **沉淀到 wiki：** [InfiniteDiffusion / Terrain Diffusion](../../wiki/entities/paper-infinite-diffusion-terrain-diffusion.md)

---

## 仓库能力（README 快照）

| 模块 | 入口 | 要点 |
|------|------|------|
| **InfiniteDiffusion 入门** | `annotated_infinite_panorama.py` | 仅依赖 SD v1.5 + infinite-tensor；生成 2048px 宽无限全景 crop |
| **地形探索 GUI** | `python -m terrain_diffusion explore <model>` | 左粗图点击 → 右高分辨率 shaded relief；可切换温度图层 |
| **查询 API** | `python -m terrain_diffusion api <model>` | 高程/气候数据查询；见 `API_README.md` |
| **Azgaar 条件** | `azgaar-to-tiff` | Fantasy Map Generator 全 JSON → 条件 GeoTIFF |
| **TIFF 导出** | `tiff-export` | 条件文件夹 → refine → **256×** 上采样 GeoTIFF |
| **训练** | `TRAINING.md` | ETOPO + WorldClim；coarse 模型可轻量重训 |

## 预训练模型

| Hugging Face ID | 分辨率 | 推荐场景 |
|-----------------|--------|----------|
| `xandergos/terrain-diffusion-30m` | 30 m/px，粗 ~7.7 km | 游戏/交互；局部细节更丰富 |
| `xandergos/terrain-diffusion-90m` | 90 m/px，粗 ~23 km | 大尺度写实世界构建 |

## 安装要点

```bash
git clone https://github.com/xandergos/terrain-diffusion
cd terrain-diffusion
pip install -r requirements.txt
# NVIDIA GPU 推荐：
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

- **Mac：** CPU only，显著更慢。
- **Windows / Linux + CUDA：** 推荐 GPU 加速。

## Minecraft Mod（terrain-diffusion-mc）

| 项 | 说明 |
|----|------|
| 平台 | Minecraft Java **1.21.1 / 1.21.11**，**Fabric** + Fabric API |
| 环境 | Server-side + Singleplayer；Windows GPU 为主（Linux/CPU 见 GitHub 说明） |
| 模型下载 | 首次启动需在线下载 **~2.5 GB** 至 `.minecraft/terrain-diffusion-models` |
| 资源 | VRAM **~1.5 GB**（offload 开启）；RAM **≥2.5 GB** 分配给 JVM |
| 世界类型 | 创建世界选 **Terrain Diffusion**；**Customize → World Scale (1–6)** |
| 探索 | `/td-explore` → 本地 Web UI（默认 `http://localhost:19801`） |
| 配置 | `config/terrain-diffusion-mc.properties`：`inference.device`、`inference.offload_models`、`explorer.port` 等 |

### World Scale 速查

- `scale=1` → **30 m/block**；`scale=2` → **15 m/block**（推荐平衡）
- 更低 scale → GPU 压力更大；更高 scale → CPU/世界高度压力更大

## 与本仓库知识的关系

| 主题 | 关系 |
|------|------|
| [Procedural Terrain Generation](../../wiki/concepts/procedural-terrain-generation.md) | 经典 Perlin/高度场程序化地形的 **学习式扩散后继** |
| [Generative World Models](../../wiki/methods/generative-world-models.md) | 生成式 **环境几何层**（高程/气候场），非操纵视频 WM |
| [Legged Gym](../../wiki/entities/legged-gym.md) | 腿式 RL 仿真中 `terrain.py` 程序化地形对照 |
| [Domain Randomization](../../wiki/concepts/domain-randomization.md) | 若将学习式地形接入 RL，仍需 DR 覆盖动力学/感知残差 |

## 对 wiki 的映射

- 主实体页：**`wiki/entities/paper-infinite-diffusion-terrain-diffusion.md`**
- 论文摘录：**`sources/papers/infinite_diffusion_terrain_diffusion_siggraph_2026.md`**

## 外部参考

- [项目页](https://xandergos.github.io/terrain-diffusion/)
- [GitHub 仓库](https://github.com/xandergos/terrain-diffusion)
- [Minecraft Mod（Modrinth）](https://modrinth.com/mod/terrain-diffusion)
- [Minecraft Mod 源码](https://github.com/xandergos/terrain-diffusion-mc)
- [arXiv:2512.08309](https://arxiv.org/abs/2512.08309)
