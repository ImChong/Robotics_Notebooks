# Evo-RL（MINT-SJTU/Evo-RL）

> 来源归档

- **标题：** Evo-RL
- **类型：** repo（LeRobot 0.4.4 分支，Apache-2.0）
- **来源：** SJTU & Evo-Tech（MINT-SJTU）
- **链接：** <https://github.com/MINT-SJTU/Evo-RL>
- **代码：** <https://github.com/MINT-SJTU/Evo-RL>
- **论文：** <https://arxiv.org/abs/2610.13345>（见 [evo_rl_arxiv_2610_13345.md](../papers/evo_rl_arxiv_2610_13345.md)）
- **数据集：** <https://huggingface.co/datasets/MINT-SJTU/RW-RL-Dataset>（见 [huggingface-co-mint-sjtu-rw-rl-dataset.md](../sites/huggingface-co-mint-sjtu-rw-rl-dataset.md)）
- **入库 / 核查日期：** 2026-10-12（浅克隆 tip `c735d69`，提交日 2026-09-15，早于论文 v1）
- **沉淀到 wiki：** [`wiki/entities/paper-evo-rl.md`](../../wiki/entities/paper-evo-rl.md)

## 仓库结构与流程（核查 README）

| 步骤 | 入口 |
|------|------|
| 遥操作 / 硬件 | `lerobot-teleoperate`、`lerobot-setup-can`（SO101、AgileX PiPER/PiPER-X） |
| 带人类接管的采集 | `lerobot-human-inloop-record`（`src/lerobot/scripts/lerobot_human_inloop_record.py`），写 `complementary_info.is_intervention` 等字段 |
| 价值函数训练 | `lerobot-value-train`，默认 `--value.type=pistar06`（`src/lerobot/values/pistar06`） |
| 价值/优势推理 | `lerobot-value-infer`，写回 value / advantage / `acp_indicator`（`--acp.n_step`、`--acp.positive_ratio`） |
| 策略训练 | `lerobot-train --acp.enable=true --acp.indicator_field=...`，指示符标签注入任务文本（dropout 0.3） |
| 下一轮 | 部署策略再用 `lerobot-human-inloop-record --policy.path=...` 采集 |

## 开放范围与边界

- 代码已开源并可跑 SO101 与 AgileX 真机流程；README 的迭代回路是“价值→优势→二值指示符→优势条件策略”，即优势条件（ACP/RECAP 式）路线。
- 论文核心的“top-10% 正优势片段选择、20 帧窗口合并、示范采样权重”在 README 与检索到的脚本中**未见对应入口**（grep 无匹配），是否已实现未核实。
- 模型权重标 “Coming Soon”；README 的 Paper 徽章仍标 Coming Soon；项目网站链接 MINT-SJTU.github.io/Evo-RL 核对时返回 404。
