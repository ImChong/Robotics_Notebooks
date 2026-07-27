# WorldScore: A Unified Evaluation Benchmark for World Generation

> 来源归档（ingest）

- **标题：** WorldScore: A Unified Evaluation Benchmark for World Generation
- **类型：** paper
- **来源：** arXiv abs / HTML；项目页、GitHub、Hugging Face Dataset / Leaderboard 交叉核对
- **原始链接：**
  - <https://arxiv.org/abs/2504.00983>
  - <https://haoyi-duan.github.io/WorldScore/>
  - <https://github.com/haoyi-duan/WorldScore>
  - <https://huggingface.co/datasets/Howieeeee/WorldScore>
  - <https://huggingface.co/spaces/Howieeeee/WorldScore_Leaderboard>
- **作者：** Haoyi Duan\*、Hong-Xing "Koven" Yu\*、Sirui Chen、Li Fei-Fei、Jiajun Wu（\* 同等贡献）
- **机构：** 斯坦福大学（Stanford University）
- **venue / 状态：** **ICCV 2025**（仓库 README）；arXiv:2504.00983
- **入库日期：** 2026-07-27
- **一句话说明：** 首个面向 **world generation** 的统一评测：把世界生成拆成带显式相机轨迹的 **next-scene** 序列，用 3000 例覆盖静/动、室内外、写实/风格化，以 **可控性 / 质量 / 动态** 十项指标汇总为 WorldScore-Static / Dynamic，统一评测 3D、4D、I2V、T2V。

## 开源核查（2026-07-27）

| 项 | 状态 |
|----|------|
| 项目页 | <https://haoyi-duan.github.io/WorldScore/> — Abstract / Dataset / Evaluation Results / Related / BibTeX；链到 arXiv、代码、HF |
| GitHub | <https://github.com/haoyi-duan/WorldScore> — **MIT**；数据集下载、`world_generators/` 适配、`worldscore/run_evaluate.py` 评测、Slurm/submitit |
| 数据集 | Hugging Face **Howieeeee/WorldScore** — `static` 2000 + `dynamic` 1000 |
| Leaderboard | Hugging Face Space **Howieeeee/WorldScore_Leaderboard**（static `index.html` + `leaderboard.csv`）；自测提交 `worldscore.json` 至 haoyiduan@princeton.edu |
| 结论 | **已开源** — 评测代码 + 数据集 + 可更新榜单齐全；评测依赖 DROID-SLAM / Grounding-SAM / SAM2 / VFIMamba 等较重 |

## 核心论文摘录（MVP）

### 1) 问题：单场景视频榜测不出「生成世界」

- **链接：** <https://arxiv.org/abs/2504.00983> §1；项目页 Overview
- **摘录要点：** VBench、WorldModelBench 等偏 **单场景** 感知质量，缺 **多场景、长序列、相机控制、3D 一致性**；且许多 3D/4D 方法需要图像条件与相机轨迹，现有 T2V 榜无法统一接入。WorldScore 把世界生成写成 next-scene 序列，统一输出为视频再打分。
- **对 wiki 的映射：**
  - [WorldScore（论文实体）](../../wiki/entities/paper-worldscore.md) — 问题定位与基准对比表。
  - [EWMBench](../../wiki/entities/ewmbench.md) — 对照：具身操纵三轴 vs 开放域 world generation 统一榜。
  - [Generative World Models](../../wiki/methods/generative-world-models.md) — 「世界生成」评测口径。

### 2) 形式化：世界规格 \((\mathcal{C},\mathcal{N},\mathcal{L})\)

- **链接：** arXiv §3.1；项目页
- **摘录要点：**
  - 当前场景 \(\mathcal{C}=\{\mathbf{I},\mathcal{P}\}\)（图像 + 文本）；下一场景文本 \(\mathcal{N}\)；布局 \(\mathcal{L}=\{\mathcal{T},\mathcal{Y}\}\)（相机轨迹矩阵序列 + 运镜文本）。
  - \(\mathbf{V}=g_{\text{world}}(w_{\text{proc}}(\mathcal{C},\mathcal{N},\mathcal{L}))\)：经模型相关预处理后生成视频，统一评测。
  - **Static：** 大运镜 + 新场景内容，测可控性与质量；**Dynamic：** 固定相机 + 场景内运动描述，测动力学。
- **对 wiki 的映射：**
  - [WorldScore](../../wiki/entities/paper-worldscore.md) — 流程总览与方法栈。

### 3) 数据集：3000 例静/动 × 写实/风格化

- **链接：** arXiv §3.2；HF Dataset card
- **摘录要点：** Static **2000**（室内外各 5 类；约 20% 为 4 场景 large world）；Dynamic **1000**（5 类运动）；每例有风格化对偶。下一场景描述由 LLM 自回归生成；8 类电影运镜随机分配。HF 拆分为 `static` / `dynamic` configs。
- **对 wiki 的映射：**
  - [WorldScore](../../wiki/entities/paper-worldscore.md) — 数据集速查。
  - [worldscore repo](../repos/worldscore.md) — `download.py` → `$DATA_PATH/WorldScore-Dataset`。

### 4) 指标：Controllability × Quality × Dynamics → Static / Dynamic

- **链接：** arXiv §3.3；项目页 Quality / Dynamics
- **摘录要点：**
  - **可控性：** Camera Ctrl（相对 GT 轨迹的尺度不变旋转/平移误差）、Object Ctrl（开集检测成功率）、Content Align（CLIPScore）。
  - **质量：** 3D Consist（DROID-SLAM 重投影）、Photo Consist（光流 AEPE）、Style Consist（Gram）、Subjective Qual（CLIP-IQA+ × Aesthetic，经人研选型）。
  - **动态：** Motion Acc / Mag / Smooth（光流区域对比、幅值、VFI 插帧平滑）。
  - 线性归一化到 0–100 后算术平均：Static = Ctrl+Quality；Dynamic = Ctrl+Quality+Dynamics；无动态的 3D 模型动力学维记 0。
- **对 wiki 的映射：**
  - [WorldScore](../../wiki/entities/paper-worldscore.md) — 十项指标表与读榜注意。

### 5) 论文结论与后续榜单

- **链接：** arXiv §4–5；HF Leaderboard（截至 2026-07-27）
- **摘录要点：** 论文评 **20** 模型：3D（WonderWorld 72.69 Static 等）强于视频的相机控制与一致性，但动力学为 0；视频主弱点是相机可控性；当时开源 CogVideoX-I2V 综合可媲美闭源 Gen-3/Hailuo；长序列与户外对视频更难。HF 榜后续扩至 **34** 行（含 Voyager、Wan2.1、WorldScape、UniWorld-View 等）；截至入库日 Static 前列含 UniWorld-View **85.53**、WorldScape-0.2(MoE) **85.13**。
- **对 wiki 的映射：**
  - [WorldScore](../../wiki/entities/paper-worldscore.md) — 评测表 + 结论。
  - [WorldScore Leaderboard](../sites/worldscore-leaderboard-hf.md) — 活榜与提交方式。

## 当前提炼状态

- [x] arXiv 摘要、§3 形式化/数据/指标、§4 观察与 §5 结论已摘录
- [x] 项目页、GitHub、HF Dataset / Leaderboard 开源核查完成
- [x] wiki 映射：`wiki/entities/paper-worldscore.md`，并与 EWMBench / 生成式世界模型 / 具身评测选型交叉引用
