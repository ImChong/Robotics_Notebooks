# NVlabs/SimFoundry

> 来源归档（ingest）

- **标题：** SimFoundry（官方实现）
- **类型：** repo
- **组织：** NVlabs（NVIDIA Research / GEAR Lab）
- **代码：** <https://github.com/NVlabs/SimFoundry>
- **Stars / Forks：** ~309 / 22（2026-09-05 核查）
- **许可证：** NVIDIA 源码 Apache-2.0；部分第三方模型/SDK 另有非商用或研究限制（见仓内 `THIRD_PARTY_LICENSES.md`）
- **默认分支：** `main`；仓创建 2026-07-15；README 记 **2026-08-14** 首发 V0 刚体/铰接管线，**2026-08-26** 发布示例场景
- **语言：** Python
- **论文：** arXiv:2606.28276 — 归档见 [sources/papers/simfoundry_arxiv_2606_28276.md](../papers/simfoundry_arxiv_2606_28276.md)
- **项目页：** <https://research.nvidia.com/labs/gear/simfoundry/> — 归档见 [sources/sites/nvidia-research-simfoundry.md](../sites/nvidia-research-simfoundry.md)
- **示例资产：** Hugging Face `nadunRanawaka1/simfoundry-assets`（dataset）
- **入库日期：** 2026-09-05
- **一句话说明：** 官方 **部分开源**：可跑 **A 重建 / B cousins / C OmniGibson 加载**；论文级 **策略训练与评测协议未随仓发布**。
- **沉淀到 wiki：** 是 → [`wiki/entities/paper-simfoundry-real2sim-scene-generation.md`](../../wiki/entities/paper-simfoundry-real2sim-scene-generation.md)

## 开源边界（步骤 2.5，2026-09-05）

| 状态 | 内容 |
|------|------|
| **已发布** | `scripts/pipeline/A_reconstruction/`（视频→OmniGibson 场景，约 13 阶段）、`B_augmentation/`（object/scene/task cousins + 任务 YAML）、`C_application/`（smoke / 加载 / 遥操作脚手架）、浏览器 light editor + OmniGibson 交互编辑器、HF 示例场景 |
| **待发布 / Coming Soon** | README 明确：**机器人数据生成、策略训练与评测** 不在本仓；**自动背景生成** 仍标 Coming Soon（仓内有可选 `2c_train_bg_splat.py`，但未当默认一键路径宣传） |
| **论文 vs 仓** | 论文写导出 **Isaac Lab** 等下游；开源默认编译到 **OmniGibson**（`s14_og/reconstructed_og_scene.json`） |

项目页与官方仓互指（README badge → GEAR 项目页 + arXiv）。以 **NVlabs/SimFoundry** 为复现入口，勿只凭论文 PDF 写「暂无代码」。

## 仓库结构（顶层）

```
SimFoundry/
├── scripts/pipeline/A_reconstruction/   # 视频 / ZED → OG 场景
├── scripts/pipeline/B_augmentation/     # cousins + 任务提议
├── scripts/pipeline/C_application/      # smoke / eval / teleop / demo 脚手架
├── scripts/interactive/                 # light editor + OG 编辑器
├── scripts/installation/                # install_everything.sh / checkpoints
├── scripts/cfg/                         # Hydra 配置
├── simfoundry/                          # 核心库
├── docs/                                # INSTALL / AGENT_INSTALL / 示例视频
└── tests/
```

## 可运行入口（对齐 README）

```bash
bash scripts/installation/install_everything.sh
bash scripts/installation/download_checkpoints.sh --default
# 可选铰接：
bash scripts/installation/install_articulate.sh

bash scripts/pipeline/A_reconstruction/run.sh \
  --scene-name my_scene --video-fpath /path/to/video.mov
# 24 GiB 卡：再加 -- s7_mesh.low_vram=true

bash scripts/pipeline/B_augmentation/run.sh --scene-name my_scene
bash scripts/pipeline/C_application/run.sh --scene-name my_scene --mode smoke-random
```

统一调度：`scripts/pipeline/run.sh A_reconstruction|B_augmentation|C_application`。

## 环境与依赖（摘要）

- Linux + NVIDIA GPU + CUDA；Mamba/Conda；`ffmpeg`；全量安装约 **250 GB** 磁盘
- Hugging Face 门控权重：SAM3、DINOv3-ViT-L、RMBG-2.0；可选 FLUX.1-Kontext
- VLM 阶段走 **Vertex AI Gemini**（`GCLOUD_PROJECT`）或 `GEMINI_API_KEY`
- 默认 2D→3D：**Hunyuan3D-2.1**（`hunyuan` env）；可选 TRELLIS.2 / Pixal3D
- 默认 mesh 生成约 **29 GiB**；24 GiB 必须 `s7_mesh.low_vram=true`

## Pipeline A 阶段编号（复现时对齐目录）

`1b` 抽帧 → `2` 深度 → `3` 地面 → `4` 世界系 → `5` 分解物体 → `6` 超分 → `7` mesh → `8` 位姿 →（可选 `9` 铰接）→ `10` 编译 → `11` sim-ready → `12` 物理沉降 → `13` USD → `14` `reconstructed_og_scene.json`。

## 对 wiki 的映射

- [SimFoundry 论文实体](../../wiki/entities/paper-simfoundry-real2sim-scene-generation.md)
- [BEHAVIOR-1K / OmniGibson](../../wiki/entities/behavior-1k.md) — 开源导出运行层
- [Sim2Real](../../wiki/concepts/sim2real.md)
- [仿真评测基础设施](../../wiki/concepts/simulation-evaluation-infrastructure.md)
