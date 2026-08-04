# DiffSHEG（JeremyCJM/DiffSHEG）

> 来源归档（仓库 README / 脚本入口要点摘录，非全文镜像）

- **标题：** DiffSHEG — A Diffusion-Based Approach for Real-Time Speech-driven Holistic 3D Expression and Gesture Generation
- **类型：** repo
- **组织 / 作者：** HKUST + IDEA — Junming Chen, Yunfei Liu, Jianan Wang, Ailing Zeng, Yu Li, Qifeng Chen
- **代码：** <https://github.com/JeremyCJM/DiffSHEG>
- **论文：** <https://arxiv.org/abs/2401.04747>（CVPR 2024）
- **项目页：** <https://jeremycjm.github.io/proj/DiffSHEG/>
- **视频：** <https://www.youtube.com/watch?v=HFaSd5do-zI>
- **许可：** BSD-3-Clause（根目录 `LICENSE`）
- **入库日期：** 2026-07-31
- **一句话说明：** CVPR 2024 官方实现：语音驱动的整体 3D 表情 + 手势联合扩散生成（UniEG-Transformer + FOPPAS 任意长实时采样）；支持 BEAT / SHOW（TalkSHOW）训练与自定义 wav 推理，权重经 Google Drive 发布。

## 开源状态（项目页 + 仓库核查，2026-07-31）

| 模块 | 状态 |
|------|------|
| 训练 / 推理代码（`runner.py`、`trainers/`、`models/`） | **已发布** |
| 自定义音频推理脚本（`inference_custom_audio_{beat,show}.sh`） | **已发布** |
| 预训练 checkpoint | **已发布**（[Google Drive](https://drive.google.com/file/d/1JPoMOcGDrvkFt7QbN6sEyYAPOOWkVN0h/view)） |
| 数据统计包（`assets/data.tar.gz`） | **已发布**（需 untar 到仓根 `data/`） |
| BEAT / SHOW 原始数据集 | **外部公开数据集**（需按 BEAT / TalkSHOW 流程自行准备，非本仓内置） |
| Blender 可视化（`assets/beat_visualize.blend`） | **已发布**（BEAT）；SHOW 可视化指引 TalkSHOW |

- **结论：已开源。** 训练、测试、自定义音频推理与 checkpoint 均可公开获取；原始 mocap/音视频数据走 BEAT / SHOW 上游渠道。
- **项目页**（<https://jeremycjm.github.io/proj/DiffSHEG/>）展示方法框架、FOPPAS、用户研究与定量对比；代码入口以本 GitHub 仓为准。

## 依赖与运行面（README 声明）

- Ubuntu 18.04 / 20.04；Python 3.9；`torch==1.13.1+cu117`（README pip 路径）
- 环境：`assets/environment.yml`（conda）或 `assets/requirements.txt` + 上述 PyTorch wheel
- 数据统计：`cd assets && tar zxvf data.tar.gz && mv data ../`
- 实现部分基于 [BEAT](https://github.com/PantoMatrix/BEAT)、[TalkSHOW](https://github.com/yhw-yhw/TalkSHOW)、[MotionDiffuse](https://github.com/mingyuan-zhang/MotionDiffuse)

## 入口速查（对齐 README）

| 路径 / 命令 | 作用 |
|-------------|------|
| `runner.py` | 统一训练 / 测试 / 自定义音频入口 |
| `inference_custom_audio_beat.sh` | BEAT 权重 + 自定义 `.wav` 推理（FOPPAS/`ddim25`） |
| `inference_custom_audio_show.sh` | SHOW（`talkshow`）权重 + 自定义 `.wav` 推理 |
| `trainers/ddpm_beat_trainer.py` / `ddpm_show_trainer.py` | BEAT / SHOW 数据集训练器 |
| `models/transformer.py` / `gaussian_diffusion.py` / `ddpm_utils.py` | UniEG Transformer + 扩散 / DDIM |
| `datasets/beat.py` / `show.py` / `extract_hubert.py` | 数据加载与 HuBERT 特征 |
| `assets/beat_visualize.blend` | BEAT：BVH + 表情 JSON → Blender 渲染 |
| `results/`（运行后生成） | 手势 / 表情输出目录 |

**自定义音频最短路径：** 下载 checkpoint → 在对应 `inference_custom_audio_*.sh` 中设置 `--test_audio_path` 为 `.wav` → 运行脚本。

## 与本仓库知识的关系

| 主题 | 关系 |
|------|------|
| [DiffSHEG 实体](../../wiki/entities/paper-diffsheg.md) | 方法与开源边界归纳 |
| [扩散运动生成](../../wiki/methods/diffusion-motion-generation.md) | 语音条件 3D 表情+手势联合生成实例 |
| [扩散模型](../../wiki/concepts/diffusion-model.md) | DDPM/DDIM + outpainting 采样 |
| [Semantic Co-Speech Gesture（PNB）](../../wiki/entities/paper-notebook-semantic-co-speech-gesture-synthesis-and-real-ti.md) | 同属共语手势；彼为语义检索→G1 跟踪，本页为人形数字资产级生成 |

## 对 wiki 的映射

- 论文摘录：[`sources/papers/diffsheg_arxiv_2401_04747.md`](../papers/diffsheg_arxiv_2401_04747.md)
- 项目页：[`sources/sites/diffsheg.md`](../sites/diffsheg.md)
- 沉淀 **[`wiki/entities/paper-diffsheg.md`](../../wiki/entities/paper-diffsheg.md)**
