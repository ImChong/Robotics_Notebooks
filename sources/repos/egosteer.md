# egosteer/egosteer

> 来源归档

- **标题：** EgoSteer（世界模型增强 VLA + 训练基建）
- **类型：** repo
- **组织 / 作者：** egosteer（PKU / PKU–PsiBot）
- **代码：** <https://github.com/egosteer/egosteer>
- **权重：** <https://huggingface.co/EgoSteer/EgoSteer-3B-Base> · <https://huggingface.co/EgoSteer/EgoSteer-3B-RealMan>
- **基座：** [`Qwen/Qwen3-VL-2B-Instruct`](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct) + [`facebook/dinov3-vitl16-pretrain-lvd1689m`](https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m)
- **数据：** 示例 WebDataset（Google Drive `example_data.zip`）；全量处理后语料待 HF 发布
- **论文：** <https://arxiv.org/abs/2607.09701>
- **项目页：** <https://egosteer.github.io/>
- **许可：** Apache-2.0
- **入库日期：** 2026-07-23
- **一句话说明：** Qwen3-VL-2B + DiT flow-matching 动作专家 + 训练-only DINOv3 世界专家；提供单机/多机 FSDP2 训练、离线评测与 WebSocket 策略服务，默认可对接 RealMan。

## 入口速查（对齐 README）

| 路径 / 命令 | 作用 |
|-------------|------|
| `bash scripts/install.sh` | 全量训练环境（conda + CUDA 12.8 + FlashAttention） |
| `bash scripts/compute_norm_stats.sh` | 从 WebDataset shards 拟合 state/action normalizer |
| `bash scripts/train_egosteer_fsdp2_single_node.sh` | 单节点 `torchrun` 训练（`train.py`） |
| `bash scripts/train_egosteer_fsdp2.sh` / `_cloud.sh` | 多节点 pdsh / 云平台训练 |
| `bash scripts/run_server.sh` | WebSocket 策略服务（`inference.yaml`） |
| `bash scripts/run_eval.sh <ckpt> <hydra_config>` | 离线动作指标 + HTML 可视化报告 |
| `docker pull egosteerai/inference-server:latest` | 推理容器；`./create_container.sh` 挂载仓与 checkpoint |

## 与本仓库知识的关系

- 论文归档：[`sources/papers/egosteer_arxiv_2607_09701.md`](../papers/egosteer_arxiv_2607_09701.md)
- 姊妹仓：[`egosmith`](./egosmith.md)（人视频策展）、[`robot-stack`](./egosteer-robot-stack.md)（真机遥操作 / DAgger 客户端）
- wiki：[`wiki/entities/paper-egosteer.md`](../../wiki/entities/paper-egosteer.md)
