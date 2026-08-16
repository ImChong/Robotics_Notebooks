# Dstate/ODEWorld

> 来源归档

- **标题：** ODEWorld（官方实现）
- **类型：** repo
- **组织 / 作者：** Dstate（Dongxiu Liu 等）
- **代码：** <https://github.com/Dstate/ODEWorld>
- **权重：** HF collection [`ldxxx/odeworld`](https://huggingface.co/collections/ldxxx/odeworld)
- **论文：** <https://arxiv.org/abs/2607.27924>
- **项目页：** <https://dstate.github.io/odeworld_website/>
- **许可：** 截至 **2026-08-16** GitHub **未挂 LICENSE**（`license: null`）
- **入库日期：** 2026-08-16
- **一句话说明：** PT-Flow 连续时间 latent 世界模型的官方推理仓：`demo_infer.py` 用 `torchdiffeq.odeint`（RK4）在 LIBERO / AgiBot 示例上做 ODE rollout、RAE 解码视频与 PCA 速度场可视化；五套 HF 预训练权重。**训练脚本未随仓发布。**

## 入口速查（对齐 README / `demo_infer.py`）

| 路径 / 命令 | 作用 |
|-------------|------|
| `conda create -n odeworld python=3.10` + `pip install torch==2.6.0 … cu124` + `requirements.txt` | 环境；依赖含 `torchdiffeq` |
| `hf download ldxxx/ODEWorld-PT-Flow-LIBERO` 等 5 个模型 → `assets/pretrained/` | 推理权重 |
| `python demo_infer.py --dataset libero` | LIBERO 示例：GT goal rollout + PCA 场；有 Goal-Predictor 时再出语言目标 rollout |
| `python demo_infer.py --dataset agibot` | AgiBot 示例（无 Goal-Predictor） |
| `--case-ids case_00` | 只跑单个 case；输出 `outputs/<dataset>/<case_id>/` |
| `models/DINOv2PTFlow.py` | PT-Flow：`latent_encode` / `delta_decouple` / `forward_vmodel` / `rollout_ode` |
| `models/DINOv2RAE.py` | DINO 特征 ↔ 图像解码 |
| `models/DINOv2GoalPred.py` | 语言 → 目标图像（仅 LIBERO 权重） |
| `models/DINOv2Latent.py` | 动力学 encoder/decoder 组件 |
| `assets/libero` / `assets/agibot` + `manifest.json` | demo PNG 与指令 |

## 与本仓库知识的关系

| 主题 | 关系 |
|------|------|
| [ODEWorld](../../wiki/entities/paper-odeworld.md) | 实体归纳：PT-Flow、连续时间预测、子目标策略 |
| [Generative World Models](../../wiki/methods/generative-world-models.md) | 连续时间 latent 生成对照离散视频扩散 |
| [V-JEPA 2](../../wiki/entities/paper-vjepa2.md) | 论文视频基线；监督方式（JVP 速度 vs JEPA 一致性）对照 |
| [Latent Imagination](../../wiki/concepts/latent-imagination.md) | ODE 积分展开 vs RSSM 离散步想象 |

## 对 wiki 的映射

- 论文摘录：[`sources/papers/odeworld_arxiv_2607_27924.md`](../papers/odeworld_arxiv_2607_27924.md)
- 项目页：[`sources/sites/odeworld-website.md`](../sites/odeworld-website.md)
- 沉淀 **[`wiki/entities/paper-odeworld.md`](../../wiki/entities/paper-odeworld.md)**
