# Zero-WAM（robbyant-research/Zero-WAM）

- **URL：** <https://github.com/robbyant-research/Zero-WAM>
- **许可：** Apache-2.0
- **语言：** Python（PyTorch / FlashAttention / FlexAttention / WebSockets / LeRobot 0.3.3）
- **配套论文：** [arXiv:2608.26103](https://arxiv.org/abs/2608.26103)（[`zero_wam_arxiv_2608_26103.md`](../papers/zero_wam_arxiv_2608_26103.md)）
- **项目页：** <https://robbyant-research.github.io/Zero-WAM/>（[`zero-wam.md`](../sites/zero-wam.md)）
- **模型与数据：** `robbyant-research/zero-wam-pretrain`、`robbyant-research/zero-wam-posttrain-robotwin`、`datasets/robbyant-research/HumanGen`
- **核对范围：** 通读 `README.md`、`INSTALL.md`、`Makefile`、`script/train.sh`、`wan_va/configs/zerowam_train_config.py`、`wan_va/train.py`、`wan_va/modules/icl_model.py`、`wan_va/mcp.py`、`wan_va/wan_va_server.py`、`evaluation/robotwin/run_icl_eval.sh` 与 `evaluation/robotwin/eval_policy_client_openpi.py`
- **通读：** 已完成

## 状态（2026-11-19 再核）

| 项 | 状态 | 说明 |
|----|------|------|
| README / 项目页 / 安装指南 | **已发布** | 明确 Python 3.10 + PyTorch 2.9.0 + CUDA 12.6 及 RoboTwin 2.0 依赖与评测协议变更 |
| 训练与模型代码 | **已开源** | `script/train.sh` -> `torch.distributed.run -m wan_va.train`；含 `WanICLTransformer3DModel`、`ICLAttentionBackend` 与 `mcp.py` |
| 推理服务与 RoboTwin 评测 | **已开源** | `evaluation/robotwin/run_icl_eval.sh` 拉起 `wan_va_server.py`（`VA_Server`）与 `eval_policy_client_openpi.py` |
| 模型权重 | **已发布** | Hugging Face 提供 `zero-wam-pretrain` 与 `zero-wam-posttrain-robotwin`（10.8B MoT） |
| HumanGen 数据集 | **部分发布** | `robbyant-research/HumanGen` 含 RoboTwin ICL 与 5 份外部预训练子集潜变量；in-house/real 数据与视频生成管线脚本未开放 |

## 实读源码核心入口与配置

1. **训练入口：** `script/train.sh` 启动 `torch.distributed.run -m wan_va.train`，由 `Trainer` 加载 `MultiICLLeRobotLatentDataset`/`MixedICLLeRobotLatentDataset` 与 `WanICLTransformer3DModel`（`icl_rope_h=24`、`drop_icl=0.1`、`droptext_target=0.4`）。
2. **IFP / MCP 损失模块：** 代码内称为 MCP（`wan_va/mcp.py`），抽取主干隐藏层 `[3, 11, 19, 29]`，接 `K=4` 个单层 Transformer 块预测跨度 `stride=2` 的未来视频 chunk（配置文件损失权重 `[0.5, 0.25, 0.15, 0.1]`，SNR shift `10`）。
3. **在线 WebSocket 服务与评测：** `VA_Server` 在 `_reset_icl` 中将人类示范编码进 `ICL_CACHE_TYPE=2` KV 缓存，`_infer_icl` 逐 chunk 去噪视频潜变量并生成动作 chunk（动作 query 在 `ICLAttentionBackend.build_self_mask` 中屏蔽对人类示范 KV 的直接注意），`_compute_icl_kv_cache` 清理预测缓存并追加真实观测缓存。
4. **RoboTwin 协议修订（`INSTALL.md`）：** `evaluation/robotwin/envs/move_stapler_pad.py`（不约束偏航角）与 `stamp_seal.py`（目标平面容限 `1 cm -> 3 cm`）覆盖上游任务文件，并对全部对比方法一致生效。

## wiki

- [`wiki/entities/paper-zero-wam.md`](../../wiki/entities/paper-zero-wam.md)
