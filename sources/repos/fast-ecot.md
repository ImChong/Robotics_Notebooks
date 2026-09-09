# Fast-ECoT（kevinDuan1/Fast-ECoT）

> 来源归档

- **标题：** Fast ECoT
- **类型：** repo
- **来源：** UCL / University of Freiburg / Cisco Research
- **链接：** <https://github.com/kevinDuan1/Fast-ECoT>
- **论文：** <https://arxiv.org/abs/2506.07639>
- **上游：** [embodied-CoT](embodied-cot.md)、[OpenVLA](openvla.md)
- **许可：** MIT
- **入库日期：** 2026-09-09
- **一句话说明：** ECoT 推理加速：thought caching、并行模块化推理、异步调度；LIBERO / Bridge / DROID 评测脚本。
- **沉淀到 wiki：** [`wiki/entities/paper-fast-ecot.md`](../../wiki/entities/paper-fast-ecot.md)

---

## 仓库入口（README）

| 组件 | 说明 |
|------|------|
| 安装 | `pip install -e .`；可选 `vllm`、`flash-attn` |
| 训练 | `vla-scripts/train.py` / `finetune.py`（LoRA on `Embodied-CoT/ecot-openvla-7b-oxe`） |
| 部署 | `vla-scripts/deploy.py` |
| LIBERO 评测 | `experiments/robot/libero/run_libero_eval.py --reasoning True --use_vllm True --async_engine True` |
| Bridge / DROID | `experiments/robot/bridge/`、`droid/` |
| 核心包 | `prismatic/`（模型）、`experiments/robot/openvla_utils.py`、`async_utils.py` |

## 开源边界（截至 2026-09-09）

- **已开源**：完整推理加速栈与评测脚本（MIT）。
- **无独立项目页**：入口为 GitHub README + arXiv。
- **权重**：沿用 HF `Embodied-CoT/*`（Llama-2 许可约束）。
- **依赖**：基于 [ECoT](embodied-cot.md) 与 OpenVLA 代码结构。
