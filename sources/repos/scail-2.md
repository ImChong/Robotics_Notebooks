# SCAIL-2（zai-org/SCAIL-2）

> 来源归档

- **标题：** SCAIL-2 — Unifying Controlled Character Animation with End-to-end In-Context Conditioning（官方实现）
- **类型：** repo
- **组织：** Z.ai（`zai-org`）· 清华大学
- **代码：** <https://github.com/zai-org/SCAIL-2>
- **项目页：** <https://teal024.github.io/SCAIL-2/>
- **论文：** <https://arxiv.org/abs/2606.10804>
- **权重：** <https://huggingface.co/zai-org/SCAIL-2>
- **License：** MIT（HF model card）
- **入库日期：** 2026-09-17
- **一句话说明：** **14B** latent 视频扩散角色动画：主分支提供 **单/多 GPU 推理**（`generate.py`）与 **SCAIL-Pose** 预处理子模块；训练见 **`sat-scail2`** 分支。
- **沉淀到 wiki：** [SCAIL-2（实体页）](../../wiki/entities/paper-scail-2.md)

## 开源状态（2026-09-17 核查）

| 项 | 结论 |
|----|------|
| 主分支 | **推理** — `generate.py`、`convert.py`、ComfyUI 集成说明 |
| `sat-scail2` 分支 | **训练** — 2026-08-06 发布；社区另有 VRAM 友好 [SCAIL-2-Tuner](https://github.com/fengjia-guo/SCAIL-2-Tuner) |
| 权重 | HF 下载 `zai-org/SCAIL-2`（含 Wan VAE、T5、FSDP checkpoint）；可转 `safetensors` |
| 预处理 | 子模块 `SCAIL-Pose`：`process_animation_aio.py`（animation e2e / pose-driven）、`process_replacement.py` |
| 可选增强 | Relighting LoRA、DPO LoRA（HF）；`prompt_enhancer.py`（Gemini，非默认依赖） |
| Python | 3.10–3.12 |

## 关键入口（README 对齐）

| 模式 | 入口 | 输入 |
|------|------|------|
| Animation（e2e） | `SCAIL-Pose/.../process_animation_aio.py --e2e_mode` → `generate.py` | ref 图 + ref_mask + driving 视频 + mask 视频 |
| Animation（pose-driven） | 同上（无 `--e2e_mode`）→ `generate.py` | 骨架渲染 driving |
| Replacement | `process_replacement.py` → `generate.py --replace_flag` | 替换区域 mask + 描述性 prompt |
| Multi-reference | README Experimental；ComfyUI PR | 多 ref 图 + 分色 mask |

## 与本仓库知识的关系

| 主题 | 关系 |
|------|------|
| [SCAIL-2 论文归档](../papers/scail2_arxiv_2606_10804.md) | 方法与评测来源 |
| [Character Animation vs Robotics](../../wiki/concepts/character-animation-vs-robotics.md) | 生成式 2D/视频角色动画，非真机 WBC |
| [Diffusion-based Motion Generation](../../wiki/methods/diffusion-motion-generation.md) | 同属扩散生成运动/视频谱系 |
| SCAIL-1（arXiv:2512.05905） | 前作：3D-consistent pose ICL；SCAIL-2 进一步去骨架中间表示 |

## 为何值得保留

- 官方可运行推理 + 训练入口；复现与 ComfyUI 生态集成的导航锚点。
- 明确 **Mask 语义**（黑/白/彩色通道）与 **Mode-Specific RoPE** 的工程约束，避免误用为 pose-only 管线。
