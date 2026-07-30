# Transformer Transformer（real-stanford/transformer-transformer）

> 来源归档

- **标题：** Transformer Transformer — Motion-Conditioned Robot Co-design
- **类型：** repo / diffusion / MuJoCo / robot co-design
- **组织：** Stanford REAL Lab / Columbia（作者归属见论文）
- **代码：** <https://github.com/real-stanford/transformer-transformer>
- **项目页：** <https://transformer-transformer.github.io/>
- **论文：** <https://arxiv.org/abs/2607.25798>
- **权重与评测数据：** <https://real.stanford.edu/transformer-transformer/>（`checkpoints.zip`、`data.zip`、`rl_policies.zip`、`blender_templates.zip`）
- **训练 Zarr：** <https://huggingface.co/datasets/hqhuy/transformer-transformer>
- **License：** MIT（`t2/model/core.py` / 部分 embedding 工具保留 Meta DiT·MAE Attribution-NonCommercial；MJX/Brax 衍生 Apache-2.0；Menagerie 资产按原目录许可）
- **入库日期：** 2026-07-30
- **一句话说明：** 官方全栈：RoboTokens、程序化设计空间、Mink/RL 数据生成、DiT 训练与 hardware_gen / ctrl 评测、CMA-ES 基线、Blender 可视化；含预训练 ckpt。
- **沉淀到 wiki：** [Transformer Transformer（论文实体）](../../wiki/entities/paper-transformer-transformer.md)

## 开源状态（核查，2026-07-30）

| 资产 | 状态 |
|------|------|
| 训练 / 推理 / 评测代码 | **已开源**（`scripts/train.py`、`evaluate_ctrl.py`、`evaluate_hardware_opt.py`、`inference.py`） |
| RoboTokens + Menagerie 资产 | **已开源**（`assets/mjcf/`、`tests/robotok`） |
| 预训练 Transformer Transformer ckpt | **已发布**（lab 服务器 zip） |
| 评测轨迹 pickle | **已发布**（`data.zip`） |
| 四足 128 RL experts | **已发布**（`rl_policies.zip`） |
| 大规模训练 Zarr | **已发布**（HF，数十～数百 GB） |
| License | MIT + 上游例外文件 |

## 仓库导航（对齐时序图节点）

| 路径 | 作用 |
|------|------|
| `docs/starter.md` | 安装、`uv sync`、下 ckpt/data、ctrl / hardware_gen 最短评测 |
| `docs/data_generation.md` | RoboTokens、Mink vs RL、CMA-ES、UMI 处理 |
| `docs/training.md` | Hydra addons：`addon_ctrl_*` / `addon_hardware_*_diffusion` |
| `scripts/train.py` | 统一训练入口（`config/train.yaml` + addons） |
| `scripts/evaluate_ctrl.py` | 跨具身控制评测（Ray 仿真 workers + GPU policy server） |
| `scripts/evaluate_hardware_opt.py` | motion→机体共设计评测（Zeroth-Order / guided_diffusion） |
| `scripts/inference.py` | 推理辅助入口 |
| `scripts/datagen_rl.py` / `train_rl*.py` | 腿式专家数据与 RL 训练 |
| `config/hardware_optimizer/{zeroth_order,guided_diffusion}.yaml` | 两种机体优化器 |
| `t2/` | DiT 模型、数据、MJX env、训练逻辑 |
| `tests/robotok` | tokenize→detokenize→simulate 烟测（约 811 passed） |

## 最短复现路径（README / starter）

1. `uv sync --extra dev`（Python 3.12，需 NVIDIA + CUDA 13 级驱动做评测/训练）。
2. `pytest tests/robotok`（CPU，无需下载）。
3. 下载 `checkpoints.zip` + `data.zip` 到仓库根。
4. 控制：`python scripts/evaluate_ctrl.py evals/ctrl@eval_fn=wheeled_bimanual ckpt_path=checkpoints/wheeled_bimanual/z4454nxj/045.pt`
5. 共设计：`python scripts/evaluate_hardware_opt.py --config-name=evaluate_hardware_opt_bimanual ckpt_path=checkpoints/wheeled_bimanual/mgoc83ra/035.pt`；换 `hardware_optimizer@eval_fn.hardware_optimizer_fn=guided_diffusion` 启用 Dynamics Self-Guidance。
6. 训练：按 `docs/training.md` 从 HF 拉对应 Zarr，改 `config/train.yaml` defaults 中的 addon。

## 与本仓库知识的关系

| 主题 | 关系 |
|------|------|
| [Shape Your Body](../../wiki/entities/paper-shape-your-body-value-gradient-design.md) | 另一条共设计：价值梯度搜连续参数 vs 本文扩散生成完整机体 |
| [扩散模型](../../wiki/concepts/diffusion-model.md) | DiT + DDIM + 奖励自引导 |
| [ALOHA](../../wiki/entities/aloha.md) | 真机抛布验证平台 |
| [跨具身迁移选型](../../wiki/queries/cross-embodiment-transfer-strategy.md) | 本文补「设计侧生成 + 同模型跨具身控制」视角 |
| [双臂操作](../../wiki/tasks/bimanual-manipulation.md) | 轮式双臂洗碗 / ALOHA 抛布 |

## 为何值得保留

- 共设计与跨具身控制共用一套可运行代码与发布权重，适合写 **源码运行时序图** 与选型对照。
- RoboTokens 提供可复用的「刚体关节机器人 → 学习就绪序列」工程参考。
