# XiangchengZhang/world-action-planner（Hugging Face）

> 来源归档（ingest）

- **标题：** Models for World Action Planner
- **类型：** site / huggingface-model
- **官方入口：** <https://huggingface.co/XiangchengZhang/world-action-planner>
- **论文：** <https://arxiv.org/abs/2607.27599>
- **代码：** <https://github.com/XiangchengZhang/world-action-planner>
- **入库日期：** 2026-08-02
- **一句话说明：** 发布动作条件世界模型与配套 diffusion policy / IDM 权重（LIBERO-90 / LIBERO-Object / robosuite 等）。

## 页面公开信息（检索自 2026-08-02）

| 资源 | 状态 |
|------|------|
| 模型卡 | `README.md`：Models for [World Action Planner](https://arxiv.org/abs/2607.27599) |
| Tags | `arxiv:2607.27599` |
| 体积量级 | API 报 usedStorage ≈ **59.6 GB**（含多套 ckpt） |
| Gated | false |

### 主要文件树（节选）

| 路径前缀 | 内容 |
|----------|------|
| `world_models/libero_90_base/` | Hydra 配置 + `checkpoints/latest.ckpt` |
| `world_models/libero_object_ft/` | Object 套件微调世界模型 |
| `world_models/robosuite_ft/` | Robosuite 微调 + prompt/neg prompt |
| `diffusion_policy/libero_90/` | `dp.ckpt` + `idm_long/short.ckpt` |
| `diffusion_policy/libero_object/` | Object 套件 DP/IDM |
| `diffusion_policy/robomimic/` | robomimic IDM |

## 对 wiki 的映射

- [`wiki/entities/paper-world-action-planner.md`](../../wiki/entities/paper-world-action-planner.md)
- [`sources/repos/world-action-planner.md`](../repos/world-action-planner.md)
- [`sources/papers/world_action_planner_arxiv_2607_27599.md`](../papers/world_action_planner_arxiv_2607_27599.md)
