# Z1hanW/CRISP-Real2Sim

> 来源归档

- **标题：** CRISP-Real2Sim（官方实现）
- **类型：** repo
- **作者：** Zihan Wang 等（CMU；Bosch RCAI 资助，致谢见论文）
- **代码：** <https://github.com/Z1hanW/CRISP-Real2Sim>
- **Stars / Forks：** ~144 / 9（2026-06）
- **论文：** <https://openreview.net/forum?id=xlr3NqxUqY>（ICLR 2026）
- **arXiv：** <https://arxiv.org/abs/2512.14696>
- **项目页：** <https://crisp-real2sim.github.io/CRISP-Real2Sim/>
- **视频数据集：** <https://drive.google.com/drive/folders/1PX8Pqzqjlh5v0Z6xt-NjzTgpugk4igoN>
- **入库日期：** 2026-05-17
- **最近复核：** 2026-06-19
- **一句话说明：** ICLR 2026 **CRISP** 官方 Python 实现：单目视频 → 平面原语场景重建 + 人–场景对齐 → `MotionTracking` 子模块 RL 训练/评估；复现命令以克隆时 upstream README 为准。
- **沉淀到 wiki：** 是 → [`wiki/methods/crisp-real2sim.md`](../../wiki/methods/crisp-real2sim.md)

## 与组织仓的区分

| 仓库 | 角色 |
|------|------|
| **[Z1hanW/CRISP-Real2Sim](https://github.com/Z1hanW/CRISP-Real2Sim)** | **主代码**：`scripts/` 重建管线、`MotionTracking/` RL、`vis_scripts/` 可视化 |
| [CRISP-Real2Sim/CRISP-Real2Sim](https://github.com/CRISP-Real2Sim/CRISP-Real2Sim) | **GitHub Pages 站点源码**（`index.html`、`scene-aware-policy.html` 等），非可复现训练代码 |

项目页与论文 BibTeX 仍托管在 `crisp-real2sim.github.io`；克隆复现应指向 **Z1hanW** 个人仓。

---

## 仓库结构（顶层）

```
CRISP-Real2Sim/
├── scripts/              # 1–8 步重建管线 + 可选 contact hallucination
├── prep/                 # SMPL/SMPL-X、demo 数据、contact 资产准备
├── setups/               # setup_crisp.sh、fetch 资产、run_demo、环境校验
├── vis_scripts/viser_m/  # 人–场景重建 Viser 可视化（可 pip install -e）
├── MotionTracking/       # CRISP 资产 → RL 训练 / eval / viser / SMPL 导出
├── run_crisp_video.sh    # 自有数据一键入口
├── assets/               # teaser 等静态资源
└── runtime_shims/        # 环境兼容 shim（sitecustomize.py）
```

默认分支 `main`；另有实验分支 `2rl`、`letsee`（以 upstream 为准）。

---

## 重建管线（scripts 1–8）

README 将端到端概括为：

> `scripts` **1–8**：`1)` 视频抽帧 → `2)` 人体 mask → `3)` 改进场景重建 → `4)` 相机后处理 → `5)` **GVHMR** → `6)` 人–场景对齐与优化 → `7)` **平面拟合** → `8)` 后处理对齐 + **bridge**；随后 **`MotionTracking`** 负责 RL 训练 / 评估 / Viser。

**数据布局约定：** 源序列放在 `*_videos` 或 `*_img` 目录；调用脚本时去掉该后缀（例如 `data/demo_videos/` → 传 `data/demo`）。

**输出目录语义：**

| 路径 | 含义 |
|------|------|
| `results/output/scene/` | CRISP 直接重建（含 `scene_mesh_sqs.urdf` 等） |
| `results/output/post_scene/` | z-up 对齐、旋转后的后处理版本，供 **bridge 进 MotionTracking** |

**可选模块：**

- **Contact Hallucination：** `scripts/0_interactvlm.sh` / `all_gv_contact.sh`（InteractVLM 类接触补全，见 `prep/README.md`）
- **NKSR 稠密表面：** `setups/setup_crisp_nksr.sh` + `vis_scripts/viser_m/run_nksr.sh`（主流程不依赖）

---

## MotionTracking 子模块

`MotionTracking/` 覆盖：

- 环境安装与 **CRISP → RL** 资产桥接（`bridge_crisp_sequence.sh` 等）
- 训练：`run_bridged_train.sh`
- 评估 / Viser：`run_bridged_eval.sh`、`run_bridged_eval_viser.sh`
- 运动导出：`run_bridged_export_motion.sh`
- 代理可视化：`agents/vis_agent.py`（checkpoint + 序列名）

工作目录需先 `cd MotionTracking`；细节以子目录 `README.md` 为准。

---

## 环境与入口（摘要）

```bash
git clone --recursive https://github.com/Z1hanW/CRISP-Real2Sim.git
cd CRISP-Real2Sim
bash setups/setup_crisp.sh
conda activate crisp
# 自有数据：
bash run_crisp_video.sh /path/to/data/demo
# 或 demo 快捷：
bash setups/run_demo.sh
```

依赖清单：`requirements.txt`、`requirements-crisp-video.txt`；快速环境说明见 `CRISP_VIDEO_ENV_RELEASE.md`。

---

## 视频数据集

Google Drive 发布自采与互联网片段（含 parkour、楼梯等），并含与 **PROX / EMDB / RICH** 相关的剪辑。作者注明：高动态片段上 HMR 仍不稳定，部分序列在 CRISP 中会失败，但仍公开以缓解 Real2Sim 管线「找干净视频」瓶颈。

---

## 与本仓库知识的关系

| 主题 | 关系 |
|------|------|
| [CRISP](../../wiki/methods/crisp-real2sim.md) | 方法归纳：平面原语、接触补全、RL 物理闭环 |
| [Sim2Real](../../wiki/concepts/sim2real.md) | Real2Sim 资产质量影响后续 sim 训练与 sim2real |
| [VideoMimic](../../wiki/entities/videomimic.md) | 项目页并排对比基线 |
| [COINS](../../wiki/entities/paper-coins-compositional-human-scene-interaction.md) | 共享 PROX 等人–场景基准生态 |

## 对 wiki 的映射

- 方法页 **[`wiki/methods/crisp-real2sim.md`](../../wiki/methods/crisp-real2sim.md)**；论文摘录见 [`sources/papers/crisp_real2sim_iclr2026.md`](../papers/crisp_real2sim_iclr2026.md)。
