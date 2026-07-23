# Robert-gyj/Ctrl-World

> 来源归档

- **标题：** Ctrl-World（官方实现）
- **类型：** repo
- **组织 / 作者：** Robert-gyj（Yanjiang Guo）
- **代码：** <https://github.com/Robert-gyj/Ctrl-World>
- **权重：** <https://huggingface.co/yjguo/Ctrl-World>
- **基座：** [`stabilityai/stable-video-diffusion-img2vid`](https://huggingface.co/stabilityai/stable-video-diffusion-img2vid) + CLIP `openai/clip-vit-base-patch32`
- **数据：** HF [`cadene/droid_1.0.1`](https://huggingface.co/datasets/cadene/droid_1.0.1)（~370G）
- **论文：** <https://arxiv.org/abs/2510.10125>
- **项目页：** <https://ctrl-world.github.io/>
- **许可：** MIT
- **入库日期：** 2026-07-23
- **一句话说明：** 基于 SVD 的动作条件多视角世界模型：提供轨迹回放、键盘交互、π₀.₅ policy-in-the-loop 与 DROID 预训练 / 下游后训练脚本；亦覆盖 VLAW 文中的 WM post-training 流程。

## 入口速查（对齐 readme.md）

| 路径 / 命令 | 作用 |
|-------------|------|
| `pip install -r requirements.txt` | 基础依赖；与 π₀.₅ 交互需另装 [openpi](https://github.com/Physical-Intelligence/openpi) |
| HF `yjguo/Ctrl-World` + SVD + CLIP | 推理 / 训练权重与编码器 |
| `python scripts/rollout_replay_traj.py …` | 用录制动作在 WM 内回放长轨迹（含 `dataset_example/droid_subset`） |
| `python scripts/rollout_key_board.py …` | 键盘控制交互 rollout |
| `python scripts/rollout_interact_pi.py` / `rollout_interact_pi_eval.py` | 与 π₀.₅ 在想象空间闭环（需 `XLA_PYTHON_CLIENT_MEM_FRACTION=0.4`） |
| `accelerate launch scripts/train_wm.py …` | DROID / 子集上训练或后训练 WM |
| `dataset_meta_info/create_meta_info.py` | 提取 latent 后写 meta + 状态/动作归一化 |

## 与本仓库知识的关系

| 主题 | 关系 |
|------|------|
| [Ctrl-World](../../wiki/entities/paper-ctrl-world.md) | 实体归纳：多视角可控 WM、评估与合成改进 |
| [Generative World Models](../../wiki/methods/generative-world-models.md) | 低维动作条件 + 策略闭环范例 |
| [Masked Visual Actions](../../wiki/entities/paper-masked-visual-actions.md) | 对照：掩码像素条件 vs 关节/位姿条件 |
| [world-models-route-03-virtual-sandbox](../../wiki/overview/world-models-route-03-virtual-sandbox.md) | 虚拟评估 + 合成数据改进沙盒 |

## 对 wiki 的映射

- 论文摘录：[`sources/papers/ctrl_world_arxiv_2510_10125.md`](../papers/ctrl_world_arxiv_2510_10125.md)
- 项目页：[`sources/sites/ctrl-world-github-io.md`](../sites/ctrl-world-github-io.md)
- 沉淀 **[`wiki/entities/paper-ctrl-world.md`](../../wiki/entities/paper-ctrl-world.md)**
