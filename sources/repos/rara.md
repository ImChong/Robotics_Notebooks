# RARA 官方仓库

> 来源归档

- **标题：** Shua-Kang/RARA
- **类型：** repo
- **代码：** <https://github.com/Shua-Kang/RARA>
- **论文：** <https://arxiv.org/abs/2610.13578>（[论文归档](../papers/rara_arxiv_2610_13578.md)）
- **项目页：** <https://shua-kang.github.io/RARA_robot/>（[站点归档](../sites/rara-github-io.md)）
- **许可：** MIT
- **入库/核查日期：** 2026-10-12（浅克隆；tip 提交 2026-10-11，约 25 个文件）
- **沉淀到 wiki：** [`wiki/entities/paper-rara-retrieval-anchored-finetuning.md`](../../wiki/entities/paper-rara-retrieval-anchored-finetuning.md)

## 仓库入口

| 路径 | 作用 |
|------|------|
| `scripts/download.sh` | 下载药瓶示范、预训练集与权重（带 `small` 参数可跳过 77 GB 预训练数据） |
| `scripts/train.sh` | `MODE=pretrain / ft / rara / coft` 训练入口 |
| `scripts/select.sh` → `rara/select.py` | 按轨迹 DTW 从预训练 pick-and-place 中选 250 条 |
| `scripts/stage1.sh` → `rara/stage1.py` | 阶段 1 对齐（3000 步，lam 1.0，mpos 4，tau 0.07，lr 1e-5，单卡 <1 h） |
| `scripts/evaluate.sh`、`scripts/eval.sh` | 最后 3 个 checkpoint × 2 个种子块 × 25 episodes；`id` / `ood` 两个划分 |
| `eval/server.py`、`eval/robotwin_client.py` | 评测服务端与 RoboTwin 客户端 |
| `robotwin/` | RoboTwin 任务配置与 `workspace_override.patch` |
| `rara/workspace.py`、`rara/dataset.py`、`rara/config/` | 训练 workspace、数据与配置 |

## 数据与权重（README 列出）

HuggingFace：`ShuaKang/robotwin_pill_rara`（50 条药瓶示范 + 250 条检索示范）、`ShuaKang/my_roboTwin2.0_training`（2000 示范、5 任务）、`ShuaKang/RARA-robotwin-pretrain`、`ShuaKang/RARA-robotwin-pill`。未逐一打开这些 HF 链接核实可下载性。

## 边界

- 仅覆盖 Diffusion Policy + RoboTwin 药瓶任务；README 的预期结果（ID：FT 44.0 / Co-FT 44.7 / RARA 59.3）与论文表内同任务数字不同，见论文归档。
- 预训练需 4 GPU、约 160 GB 主机内存（README 自述）。
