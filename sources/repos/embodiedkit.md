# Embodied-Arcadia/EmbodiedKit

> 来源归档

- **标题：** EmbodiedKit（Arcadia 官方实现）
- **类型：** repo
- **组织 / 作者：** Embodied-Arcadia（浙江大学 / 宇树等，Arcadia 论文配套）
- **代码：** <https://github.com/Embodied-Arcadia/EmbodiedKit>
- **论文：** <https://arxiv.org/abs/2512.00076>
- **项目页：** 无独立站点（仓 homepage 为空）
- **License：** **未声明**（截至 2026-08-16）
- **入库日期：** 2026-08-16
- **一句话说明：** Arcadia 公开仓：Isaac 侧 VLN/VLA 数据生成 + Qwen 系训练/评测脚本可辨识；根 README 仍是 TODO，探索/3DGS/反馈闭环与权重未发布。

## 开源状态（核查 2026-08-16）

| 项 | 状态 |
|----|------|
| 仓可见性 | 公开；约 29★ / 0 fork；`created` 2025-11-17，`pushed` 2025-11-19 |
| 根 README | **占位** — 只列环境指南 / Quickstart / 重构三项 TODO |
| 可运行入口 | **部分** — 见下表；依赖本机 Isaac Sim 与大显存 GPU |
| 权重 / 数据 | **未挂** Hugging Face 或 Release |
| 论文闭环 | 探索 + Nvblox、Gaussian-splat 重建、Sim-from-Real 写回 **不在仓内** |
| License | **无** |

## 入口速查（对齐子目录 README）

| 路径 / 命令 | 作用 |
|-------------|------|
| `vln_data_generate/main_controller.py` | 批处理：遍历 USD 场景 → 调 Isaac `python.sh` |
| `vln_data_generate/path_generator.py` | NavMesh + 起终点采样，写 `generated_paths.json` |
| `vln_data_generate/robot_navigator.py` | G1 沿路径录 RGB / 位姿 / 动作（G1 USD **硬编码**） |
| `vla_data_generate/main_generator.py` / `run_simple.py` | Franka Lula RRT 生成操作轨迹 |
| `vla_data_generate/convert_to_rlds.py` | 原始输出 → RLDS / TFRecord |
| `vln_train/scripts/finetune_lora_vision.sh` | Qwen2.5-VL LoRA（LLaVA JSON） |
| `vln_train/scripts/eval_train.sh` | 合并 LoRA 后离线评测 |
| `vla_train/openqwenvla_pretrain.py` | RLDS 预训练（文档：≥80 GB） |
| `vla_train/openqwenvla_finetuning.py` | RLDS 微调 |
| `vla_train/experiments/libero/run_libero_eval.py` | LIBERO 评测 |
| `scene_replace/replace.py` | InternUtopia 相似资产替换 + 碰撞检查 |

**依赖提示：** VLN 生成写明 **Isaac Sim 5.0.0**；VLA 生成写 **Isaac Sim 4.5+**；VLA 训练文档钉 **H20 + CUDA 12.4**。场景资产、G1 USD、InternUtopia 库均需自备。`vln_train` README 混有上游 Qwen-VL 微调 fork 与「Qwen-VLN」补丁，不是端到端 VLN-CE-Isaac 一键脚本。

## 与本仓库知识的关系

| 主题 | 关系 |
|------|------|
| [Arcadia](../../wiki/entities/paper-arcadia.md) | 论文实体：四段闭环与开源边界 |
| [视觉–语言导航](../../wiki/tasks/vision-language-navigation.md) | 仿真数据与 Qwen-VL 导航微调 |
| [VLA](../../wiki/methods/vla.md) | OpenQwenVLA + LIBERO |
| [VLN 四范式复现](../../wiki/overview/vln-open-source-repro-paradigms.md) | **不能**当新手可跑通栈 |

## 对 wiki 的映射

- 论文摘录：[`sources/papers/arcadia_arxiv_2512_00076.md`](../papers/arcadia_arxiv_2512_00076.md)
- 沉淀 **[`wiki/entities/paper-arcadia.md`](../../wiki/entities/paper-arcadia.md)**
