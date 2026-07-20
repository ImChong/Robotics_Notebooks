# Chaoqi-LIU/oat（Ordered Action Tokenization）

> 来源归档（repo）

- **标题：** OAT: Ordered Action Tokenization — 官方实现
- **类型：** repo
- **URL：** <https://github.com/Chaoqi-LIU/oat>
- **姊妹仓：** [Chaoqi-LIU/praxis-vla](https://github.com/Chaoqi-LIU/praxis-vla)（作者提及的 VLA / token co-training 扩展线）
- **项目页：** <https://ordered-action-tokenization.github.io/>
- **论文：** [arXiv:2602.04215](https://arxiv.org/abs/2602.04215) — 归档见 [`sources/papers/oat_ordered_action_tokenization_arxiv_2602_04215.md`](../papers/oat_ordered_action_tokenization_arxiv_2602_04215.md)
- **入库日期：** 2026-07-20
- **一句话说明：** Harvard / Stanford 团队开源的 OAT 训练与评测栈：`oat/tokenizer` + LIBERO 数据管线 + `scripts/run_workspace.py` / `eval_policy_sim.py`；支持 `uv` 与 micromamba。

## 工程入口（README 对齐）

| 步骤 | 入口 | 说明 |
|------|------|------|
| 安装 | `uv sync && uv pip install -e .` 或 `conda_env.yaml` | 需 `--recurse-submodules` 拉取 `third_party/LIBERO` |
| 数据 | `scripts/convert_libero_dataset.py` → `compose_libero_multitask_dataset.py` | 亦可直接下 HF `chaoqi-liu/libero10_N500.zarr` |
| 训练 / 工作区 | `scripts/run_workspace.py` | 配置在 `oat/config` |
| 仿真评测 | `scripts/eval_policy_sim.py` | LIBERO 等 |

## 开源状态（2026-07-20）

- **已开源：** 训练 / 评测代码与 LIBERO 适配脚本（MIT 许可见仓库 LICENSE）。
- **权重：** 以 README / HF 发布为准；入库时以代码复现路径为主。

## 关联

- Wiki：[`wiki/entities/paper-oat-ordered-action-tokenization.md`](../../wiki/entities/paper-oat-ordered-action-tokenization.md)
- 项目页站点归档可后续补 `sources/sites/`；当前以 repo + 论文为主。
