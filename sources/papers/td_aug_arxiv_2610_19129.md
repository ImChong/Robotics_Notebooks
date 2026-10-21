# Imagining Goals with Geometric Horizon Models for Policy Generalization（arXiv:2610.19129）

> 来源归档（ingest）

- **标题：** Imagining Goals with Geometric Horizon Models for Policy Generalization
- **arXiv：** <https://arxiv.org/abs/2610.19129>（cs.LG / cs.AI，2026-10-13 提交）
- **代码：** <https://github.com/td-aug/td-aug>
- **项目页：** <https://td-aug.github.io/>
- **数据：** <https://huggingface.co/datasets/daniellawson9999/tdaug-datasets>
- **作者：** Daniel Lawson, Dhruv Sreenivas, Adriana Hugessen, Khimya Khetarpal, Glen Berseth
- **单位：** Mila；Khetarpal 另属 Université de Montréal、Google DeepMind
- **入库日期：** 2026-10-17
- **来源线索：** 具身智能小站 2026-10-17 文章（<https://mp.weixin.qq.com/s/I4XP5JSG9QNmm4hd1nk2JQ>）
- **一句话说明：** GHM 采样可达未来状态作想象目标，增广目标条件 BC，提升离线拼接泛化。

## 开源状态（步骤 2.5）

- **项目页核查（2026-10-17）：** 列出代码仓库与 HF 数据集链接；未说明是否定稿。
- **仓库核查（2026-10-17）：** MIT；`agents/`、`data_gen_scripts/`、`ogbench/`、`utils/`，入口 `main.py`、`main_gen.py`、`main_tdflow_gcrl.py`；README 含 conda / uv 安装、数据下载、示例命令；1 次提交，无 coming soon 提示。
- **结论：** **已开源**。本次未在本地运行。见 [仓库归档](../repos/td-aug.md)。

## 摘录 1：方法

- 步骤：（可选）表征学习 → 训练 GHM（flow matching + MCHC-β）→ 以 α 混合想象目标与真实未来状态训练目标条件流匹配 BC。α = 1 且样本精确时等价于一步策略改进（Theorem 1）。

**对 wiki 的映射：** [`paper-td-aug`](../../wiki/entities/paper-td-aug.md)「核心原理」。

## 摘录 2：结果

- Pointmaze stitch 平均 GCFBC 31 → TD-Aug 79；visual-antmaze 29 → 56（HIQL 69）；操作文本称 13 → 52，逐环境表抓取时列头不清（项目页平均数与 HTML 不一致）。

**对 wiki 的映射：** 同页「实验与评测」。

## 摘录 3：局限

- MCHC-β 有偏；TD²-β 样本弥散；潜空间自举困难；操作用 oracle 目标；扩展到高维部分可观测机器人未解决。

**对 wiki 的映射：** 同页「局限与风险」。

## 交叉链接

- 项目页归档：[td-aug-project.md](../sites/td-aug-project.md)
- 仓库归档：[td-aug.md](../repos/td-aug.md)
