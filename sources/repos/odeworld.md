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
- **一句话说明：** PT-Flow 连续时间 latent 世界模型的官方推理仓：`demo_infer.py` 用 `torchdiffeq.odeint`（RK4）在 LIBERO / AgiBot 示例上做 ODE rollout、RAE 解码视频与 PCA 速度场可视化；五套 HF 预训练权重。截至 2026-11-13 已含 RAE、PT-Flow 与 Goal-Predictor 训练入口，旧结论见历史日期。

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
| `models/DINOv2GoalPred.py` | 语言 → 目标 DINO 特征（仅 LIBERO 权重；图像由 RAE 解码） |
| `models/DINOv2Latent.py` | 动力学 encoder/decoder 组件 |
| `assets/examples/libero` / `assets/examples/agibot` + `manifest.json` | demo PNG 与指令 |

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

## 2026-11-13 静态源码复核

- **版本：** `fc1a7278d873d28726e26d8b3f71c89ac99ea80a`。
- **核对范围：** README 全文；`demo_infer.py`；`DINOv2PTFlow.py` 的 Imgoal forward/rollout 路径；`DINOv2Latent.py`、`DINOv2GoalPred.py`；`train_dinov2ptflow.py` 与 LIBERO shell 入口。未执行训练、下载权重或做 GPU 性能复现。
- Demo 从 `assets/examples/<dataset>/manifest.json` 读图，加载 PT-Flow/RAE，LIBERO 另加载 goal predictor；后者返回目标特征，并非 RGB。`rollout_ode` 用 RK4，速度乘 `max_time_length`，先解码回 DINO，再 RAE 分块出帧。
- 当前 README 有数据 schema（LIBERO 7D action / 9D proprio、AgiBot 16D）及三个训练阶段。`scripts/train_dinov2ptflow_libero.sh` 由 torchrun 启动八进程，`max_time_length=50`，显式 `--disable_encoder_detach`；这与类默认 detach 不同，须保留配置边界。
- 训练入口构建 `ImgVelEngine.build_vel_dataloader` 和 `Dinov2PTflowImgoal`，包装 DDP，前向返回重建/JVP 速度损失，反传、裁剪、AdamW、分组 LR schedule，保存模型和优化器。未把它称作已公开完整 X-VLA 控制实现。
- 仓树未找到独立 LICENSE；模型类 Hub metadata 标注 apache-2.0，适用范围仍需核对，不默认整仓许可。
