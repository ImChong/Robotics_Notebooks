# bytedance/IRASim

> 来源归档

- **标题：** IRASim（官方实现）
- **类型：** repo
- **组织：** ByteDance
- **代码：** <https://github.com/bytedance/IRASim>
- **License：** Apache-2.0
- **论文：** <https://arxiv.org/abs/2406.14540>
- **项目页：** <https://gen-irasim.github.io/>
- **数据 / 权重：** CDN（`scripts/download.sh`）+ HF [`fangqi/IRASim`](https://huggingface.co/datasets/fangqi/IRASim)
- **入库日期：** 2026-07-27
- **一句话说明：** Fine-grained trajectory-to-video 世界模型官方仓：安装、数据集/checkpoint 下载、Frame-Ada 训练、短/长视频评测与 Language-Table 键盘交互应用。

## 开源核查（2026-07-27）

| 项 | 状态 |
|----|------|
| 代码 | **已开源** · Apache-2.0 |
| 训练 / 评估脚本 | 可运行（`main.py`、`evaluate/`） |
| 数据与 checkpoints | RT-1 / Bridge / Language-Table 公开下载 |
| 交互 demo | `application/languagetable.py` |

## 入口速查（对齐 README）

| 路径 / 命令 | 作用 |
|-------------|------|
| `bash scripts/install.sh` | 环境安装 |
| `bash scripts/download.sh` | 拉取数据与 checkpoints |
| `python3 application/languagetable.py` | 键盘控制 Language-Table 交互生成 |
| `python3 main.py --config configs/train/rt1/frame_ada.yaml` | 单卡训 Frame-Ada |
| `torchrun … main.py --config configs/train/rt1/frame_ada.yaml` | 多卡训练 |
| `python3 evaluate/evaluation_short_script.py` | 短轨迹评测 |
| `bash scripts/generate_long_video_rt1_frame_ada.sh` | 长视频自回归生成 |

## 与本仓库知识的关系

| 主题 | 关系 |
|------|------|
| [IRASim](../../wiki/entities/paper-irasim.md) | 实体归纳 |
| [Generative World Models](../../wiki/methods/generative-world-models.md) | 动作条件视频 WM |
| [world-model-physics-fidelity-outputs](../../wiki/overview/world-model-physics-fidelity-outputs.md) | 未来视频输出族代表 |
| [Masked Visual Actions](../../wiki/entities/paper-masked-visual-actions.md) | 同属视频沙盒 / 策略评估轴 |

## 对 wiki 的映射

- 论文：[`sources/papers/irasim_arxiv_2406_14540.md`](../papers/irasim_arxiv_2406_14540.md)
- 项目页：[`sources/sites/gen-irasim-github-io.md`](../sites/gen-irasim-github-io.md)
- 沉淀 **[`wiki/entities/paper-irasim.md`](../../wiki/entities/paper-irasim.md)**
