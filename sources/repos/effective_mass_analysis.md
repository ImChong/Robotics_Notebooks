# effective_mass_analysis（RAI · AthenaZero 有效质量复现）

> 来源归档（ingest 配套仓库）

- **标题：** Effective Mass Analysis
- **类型：** repo / analysis / MuJoCo / hardware-characterization
- **组织：** RAI Institute（`rai-opensource`）
- **代码：** <https://github.com/rai-opensource/effective_mass_analysis>
- **许可：** MIT（各机器人 MJCF 子目录另有 LICENSE）
- **配套论文：** [AthenaZero SciRob aee1868](../papers/athenazero_scirobotics_aee1868.md)（DOI [10.1126/scirobotics.aee1868](https://doi.org/10.1126/scirobotics.aee1868)）
- **数据：** [Zenodo 10.5281/zenodo.21939225](https://doi.org/10.5281/zenodo.21939225)（冲击/刚度/Fig.5–6）
- **入库日期：** 2026-09-17
- **一句话说明：** 复现 AthenaZero 与 FR3 / iiwa14 / UR5e / WAM **有效质量 belted ellipsoid** 对比；提供 **简化 AthenaZero MJCF** + 各臂 `params/*.yaml`（转子惯量、减速比）；**不是** 真机棒球控制栈。

## 开源边界

| 有 | 无 |
|----|-----|
| `plot-inertia-ellipse` CLI | 完整 CAD / 制造文件 |
| 简化 AthenaZero + 四基线 MJCF | onboard 控制器 / ROS2 栈 |
| 论文 Fig.5–6 数据（Zenodo） | 抛接学习代码（见 arXiv:2608.26800） |

## 入口速查（README）

| 命令 | 作用 |
|------|------|
| `uv sync` | 安装依赖 |
| `uv run plot-inertia-ellipse --robot fr3` | 绘制 FR3 有效质量椭圆（links dashed / links+actuators solid） |
| `uv run plot-inertia-ellipse --params path/to.yaml` | 自定义机器人参数 |
| 内置机器人名 | `fr3`, `iiwa14`, `ur5e`, `wam`；AthenaZero 见 `params/` + 简化 MJCF |

## 参数文件要点

- `joints`: 关节名 → `[rotor_inertia, gear_ratio]`；反射惯量 = rotor × ratio²
- AthenaZero 腕部 **并联** → 减速比 **构型相关**；README 给 workspace 中性 `[0,0]` 值

## 关联资料

- 论文摘录：[`sources/papers/athenazero_scirobotics_aee1868.md`](../papers/athenazero_scirobotics_aee1868.md)
- 博客归档：[`sources/sites/rai-athenazero-blog.md`](../sites/rai-athenazero-blog.md)
- wiki 实体：[`wiki/entities/paper-athenazero.md`](../../wiki/entities/paper-athenazero.md)
