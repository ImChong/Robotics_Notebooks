# ParcelStow（coenwerem/parcelstow）

- **URL：** <https://github.com/coenwerem/parcelstow>
- **组织：** coenwerem
- **许可：** Apache-2.0
- **关联论文：** [parcelstow_arxiv_2609_01453](../papers/parcelstow_arxiv_2609_01453.md)
- **数据集：** [cenwerem/parcelstow](https://huggingface.co/datasets/cenwerem/parcelstow) — 归档见 [cenwerem-parcelstow](../datasets/cenwerem-parcelstow.md)

## 一句话说明

Isaac Lab 专家–学习器时间鲁棒性评测台：Unitree G1（腰 + 右臂 + L6 五指）包裹插入 / 直立放置 / 键控插销；稳定标签 `v1.0.0` 对应 arXiv v1 包裹插入。

## 运行入口（2026-09-03）

| 路径 | 作用 |
|------|------|
| `scripts/run_task.py --task parcel\|upright\|peg` | 跑脚本专家 |
| `scripts/evaluate.py --task … --actor expert\|act` | 统一评测接口 |
| `scripts/reproduce.py all-tasks` | **CPU、无 GPU**：从 `data/records/` 重算表图 |
| `scripts/download_artifacts.py --task parcel` | 拉 HF 示范与 ACT checkpoint |
| `source/parcelstow/` | Isaac Lab 扩展 |
| `docs/TASK_SPEC.md` | 包裹插入冻结科学选择 |

依赖：Isaac Lab 0.54.2 / Isaac Sim 5.1.0（仿真）；记录复现只需 NumPy + Matplotlib。

## 交叉链接

- [ParcelStow 论文实体](../../wiki/entities/paper-parcelstow.md)
- [HF 数据集](../datasets/cenwerem-parcelstow.md)
