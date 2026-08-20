# instant-episode-repetition（IER 官方实现）

> 来源归档

- **标题：** Instant Episode Repetition (IER)
- **类型：** repo
- **链接：** <https://github.com/UoA-CARES/instant-episode-repetition>
- **论文：** <https://arxiv.org/abs/2608.17347>（RLC 2026）
- **机构：** 奥克兰大学（University of Auckland）/ CARES Robot Learning Team
- **入库日期：** 2026-08-20
- **一句话说明：** IER 与 SAC/TD3 的官方实现：`train.py` + `configs/ier/*.yaml`；核心交互逻辑在 `train_loops/ier/`。
- **沉淀到 wiki：** [`wiki/entities/paper-instant-episode-repetition.md`](../../wiki/entities/paper-instant-episode-repetition.md)

---

## 入口与结构

| 路径 | 作用 |
|------|------|
| `train.py run --config configs/ier/<file>.yaml` | 主训练 CLI |
| `train_loops/ier/` | IER 交互环（探索 / 策略 / 重复三模式） |
| `algorithms/base/`、`algorithms/repetition/` | SAC、TD3 与 IER 包装 |
| `configs/ier/` | 环境、种子、RN、算法 YAML |
| `memory/`、`networks/`、`environments/` | replay、网络与环境接口 |

## 复现要点

- Python 3.10 推荐；Conda 或 venv。
- RN=0 即标准 RL；RN∈{1,…,7} 对应论文消融。
- 支持 MuJoCo、DMC 与论文所述真机任务配置（以仓库 YAML 为准）。
