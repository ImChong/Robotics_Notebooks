# Wan-Video/Wan2.1（及 Wan2.2 升级线）

> 来源归档

- **标题：** Wan 开源视频基础模型实现
- **类型：** repo
- **组织 / 作者：** Wan-Video / Wan-AI（Alibaba）
- **代码：** <https://github.com/Wan-Video/Wan2.1>（技术报告主仓）；升级 <https://github.com/Wan-Video/Wan2.2>
- **权重：** <https://huggingface.co/Wan-AI/>；ModelScope `Wan-AI`
- **论文：** <https://arxiv.org/abs/2503.20314>
- **站点：** <https://wan.video>
- **许可：** Apache-2.0
- **入库日期：** 2026-07-23
- **一句话说明：** 统一 `generate.py` 入口的开源视频 DiT 套件：T2V（1.3B/14B）、I2V、FLF2V、VACE 等；Wan2.2 增加 MoE、美学数据与 TI2V-5B。机器人侧 [Wan-Move](../../wiki/entities/paper-wan-move.md) / [Masked Visual Actions](../../wiki/entities/paper-masked-visual-actions.md) 等在此族上做可控微调。

## 入口速查（对齐 Wan2.1 README）

| 路径 / 命令 | 作用 |
|-------------|------|
| `pip install -r requirements.txt` | 安装依赖 |
| `huggingface-cli download Wan-AI/Wan2.1-T2V-14B …` 等 | 拉对应任务权重 |
| `python generate.py --task t2v-14B\|t2v-1.3B\|i2v-14B\|… --ckpt_dir …` | 单卡生成 |
| `--offload_model True --t5_cpu` | 降显存（小模型友好） |
| `torchrun … --dit_fsdp --t5_fsdp --ulysses_size …` | 多卡 xDiT / FSDP |
| Diffusers / ComfyUI | 社区推理集成（README 已列） |

## Wan2.2 相对 2.1（README 口径）

- MoE：按时间步划分专家，扩容不增同成本
- 美学标签数据 + 更大图/视频规模
- TI2V-5B：16×16×4 VAE，720P@24fps，可消费级卡

## 与本仓库知识的关系

| 主题 | 关系 |
|------|------|
| [Wan](../../wiki/entities/paper-wan-video.md) | 基础模型实体页 |
| [Wan-Move](../../wiki/entities/paper-wan-move.md) | 轨迹可控微调（I2V-14B） |
| [Masked Visual Actions](../../wiki/entities/paper-masked-visual-actions.md) | Wan2.2-Fun-Control + LoRA |
| [Generative World Models](../../wiki/methods/generative-world-models.md) | 视频先验上游 |

## 对 wiki 的映射

- 论文摘录：[`sources/papers/wan_video_arxiv_2503_20314.md`](../papers/wan_video_arxiv_2503_20314.md)
- 站点：[`sources/sites/wan-video.md`](../sites/wan-video.md)
- 沉淀 **[`wiki/entities/paper-wan-video.md`](../../wiki/entities/paper-wan-video.md)**
