# lingbot-world-v2

> 来源归档（ingest · 官方代码仓）

- **标题：** LingBot-World 2.0 / LingBot-World-Infinity
- **类型：** repo
- **机构：** 蚂蚁灵波（Robbyant）
- **链接：** <https://github.com/robbyant/lingbot-world-v2>
- **项目页：** <https://technology.robbyant.com/lingbot-world-v2>
- **论文：** <https://arxiv.org/abs/2607.07534>
- **权重：** [HF 集合 robbyant/lingbot-world-v2](https://huggingface.co/collections/robbyant/lingbot-world-v2)
- **许可：** CC BY-NC-SA 4.0
- **基座：** [Wan2.2](https://github.com/Wan-Video/Wan2.2)
- **入库日期：** 2026-09-10
- **一句话说明：** 因果交互世界模型推理栈：`generate.py` 分块 causal 推理 + KV cache；14B/1.3B 多 variant；需 torch≥2.4 + flash-attn。

## 模型变体（2026-09-10）

| 权重 | 类型 | 规模 |
|------|------|------|
| `lingbot-world-v2-14b-causal-fast` | causal-fast（蒸馏少步） | 14B |
| `lingbot-world-v2-14b-causal-pretrain` | causal-pretrain | 14B |
| `lingbot-world-v2-14b-bid` | bidirectional | 14B |
| `lingbot-world-v2-1.3b-causal-fast` | causal-fast | 1.3B |
| `lingbot-world-v2-14b-causal-fast-diffusers` | Diffusers 封装 | 14B |

## 推理入口

- **CLI：** `torchrun ... generate.py --task i2v-A14B --infer_mode causal_fast --ckpt_dir ... --image ... --action_path ... --frame_num ...`
- **脚本：** `run_fast.sh`
- **依赖：** `requirements.txt` + `flash-attn`；多卡 FSDP（`--dit_fsdp --t5_fsdp`）

## 开源状态

- **已开源**：代码 + 权重 + 示例 `examples/`（截至 2026-09-10 README News 四项 TODO 均完成）

## 对 wiki 的映射

- [paper-sa-2607-07534-infinite-worlds-with-versatile-interactions-ling](../../wiki/entities/paper-sa-2607-07534-infinite-worlds-with-versatile-interactions-ling.md)
