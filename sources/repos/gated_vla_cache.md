# Gated VLA-Cache 官方仓库（zjw4321/gated-vla-cache）

> 来源归档（repo）

- **标题：** Neural Introspection Gating for Adaptive KV-Cache Reuse in Vision-Language-Action Models
- **类型：** repo
- **链接：** <https://github.com/zjw4321/gated-vla-cache>
- **arXiv：** <https://arxiv.org/abs/2608.10824>
- **项目页：** <https://zjw4321.github.io/neural-introspection-gating-page/>
- **License：** Apache-2.0（基于 `siyuhsu/vla-cache` 提交 `a490988`，上游包括 MIT 协议的 `openvla/openvla` 与 `moojink/openvla-oft`）
- **入库日期：** 2026-11-14
- **通读：** 已完成
- **核对范围：** `README.md` (L1–131)、`NOTICE`、`openvla/run.sh`、`openvla/experiments/robot/openvla_utils.py` (L375–439)、`openvla/prismatic/extern/hf/modeling_prismatic.py` (L505–550)、`openvla/experiments/robot/libero/run_libero_eval.py` (L195–260)、`openvla-oft/experiments/robot/openvla_utils.py` (L755–835)、`openvla-oft/prismatic/extern/hf/modeling_prismatic.py` (L880–940)
- **一句话说明：** 基于 VLA-Cache 分叉实现 Gated VLA-Cache，分别提供 `openvla/` 与 `openvla-oft/` 两套代码与 LIBERO 评测入口，通过 `--use_vla_cache True --action_margin_threshold <θ_m>` 启用基于 top-1/top-2 logit margin 的 KV-Cache 自省门控。
- **沉淀到 wiki：** [`wiki/entities/paper-neural-introspection-gating.md`](../../wiki/entities/paper-neural-introspection-gating.md)

## 开源状态

- **已开源（Apache-2.0）：** 项目页（`sites/nig.html`）直接链接至 `https://github.com/zjw4321/gated-vla-cache`，包含完整推理门控修改与 LIBERO 评估脚手架。
- **未直接附带模型权重：** 需使用 `openvla/vla_cache_scripts/download_model_local.py` 或 `openvla-oft/vla_cache_scripts/download_model_local.py` 将 OpenVLA / OpenVLA-OFT 官方权重下载到本地 `checkpoints/` 目录（启动时脚本会把仓库内修改后的 `modeling_prismatic.py` 覆盖进本地 checkpoint 目录，故不能直接传 HF Hub ID）。

## 仓库结构与关键改造（来自 `NOTICE` 与源码）

| 目录 / 文件 | 作用 | 核心实现细节 |
|---|---|---|
| `openvla/` / `openvla-oft/` | 双骨干独立子目录 | 分别对应 VLA-Cache 的 `vla-cache-openvla` 与 `vla-cache-openvla-oft` 分支，需分建 conda 环境（如 PyTorch 2.2.0 或 Blackwell GPU 用 2.7.1 + `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1`） |
| `*/prismatic/extern/hf/modeling_prismatic.py` | 模型侧 margin 计算 | OpenVLA 在 `predict_action()` 中遍历 `results.scores` 计算 `action_top12_margin`；OpenVLA-OFT 在 `_regression_or_discrete_prediction()` 中从 `language_model_output.logits` 提取 `action_top12_margin` 并写入 `last_caches` |
| `*/experiments/robot/openvla_utils.py` | 推理门控调度 | `get_action()` 读取上一步 `last_caches.get("low_margin", False)`：若为 `True` 则置空 `prompt_cache`（新建 `DynamicCache()`）走全量前向；否则执行 `find_static_patches`（OpenVLA 设 `top_k=120`；OFT `top_k=150`）→ `task_relevant_selection` → `get_layer_mask_schedule`；动作返回后根据 `action_top12_margin < cfg.action_margin_threshold` 更新 `last_caches["low_margin"]` |
| `*/experiments/robot/libero/run_libero_eval.py` | 评测入口 | 新增 CLI 参数 `--action_margin_threshold`（OpenVLA 论文默认 `0.65`，OpenVLA-OFT 默认 `0.50`）与 `openvla/run.sh` 示例 |

## 复现提示（来自 `README.md`）

- **OpenVLA-OFT post-paper 索引修复（`main` vs `iros2026`）：** 论文实验（对应 git tag `iros2026`）中 OpenVLA-OFT 误从序列前端切片动作 token logits，导致复用变短序列时门控位置偏移或漏触发；`main` 分支在 `_regression_or_discrete_prediction` 改为从末尾切片 `[:, -(ACTION_DIM * NUM_ACTIONS_CHUNK) - 2 : -2, :]`。在 RTX 5090 / LIBERO-Spatial 实测上，门控触发率从 `14.5%` 升至 `33.7%`（算力 `3.25 → 3.44 TFLOPs/step`），成功率在波动范围内（`98.4%` vs `98.0%`）。
