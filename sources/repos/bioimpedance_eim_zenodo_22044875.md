# Bioimpedance EIM — Zenodo 补充数据与图表复现（Science Robotics 2026）

> 来源归档

- **标题：** Bioimpedance meets Biomechanics: Wearable Electrical Impedance Myography Encodes Fascicle and Activation Dynamics — Supplemental Data
- **类型：** repo / Zenodo artifact / dataset / analysis notebook
- **链接：** <https://zenodo.org/records/22044875>
- **DOI：** <https://doi.org/10.5281/zenodo.22044875>
- **许可证：** CC BY 4.0
- **发布：** 2026-09-23
- **论文：** [bioimpedance_eim_scirobotics_2026.md](../papers/bioimpedance_eim_scirobotics_2026.md)
- **入库日期：** 2026-09-25
- **一句话说明：** 官方 **图表与定量结果复现包**：`Manuscript_Data.xlsx`（按图/分析分 sheet 的处理后对齐数据）+ `Manuscript_Code.ipynb`（读 Excel 重绘手稿图）；**非**实时可穿戴采集或控制代码仓。
- **开源状态：** **部分开源**（数据 + 离线分析；无 GitHub 镜像）

---

## 包内容

| 文件 | 说明 |
|------|------|
| `Manuscript_Data.xlsx` | 手稿与补充图对应的 **处理后、时间对齐** 表格数据；受试者代号如 `YA02` |
| `Manuscript_Code.ipynb` | Jupyter 笔记本：按 sheet 加载并复现各图 |
| `README.md` | 依赖与运行说明 |

## 软件依赖（Zenodo README）

- Python **3.11.10**（README 标注）
- `jupyter`, `numpy`, `pandas`, `matplotlib`, `seaborn`, `scipy`, `statsmodels`, `pingouin`, `openpyxl`

```bash
python -m pip install jupyter numpy pandas matplotlib seaborn scipy statsmodels pingouin openpyxl
```

## 复现步骤

1. 将 Zenodo 三个文件下载到 **同一目录**。
2. `jupyter lab` 打开 `Manuscript_Code.ipynb`（README 中亦出现 `Manuscript_Figure_Reproduction.ipynb` 别名表述，以 Zenodo 实际文件名为准）。
3. 重启 kernel，自上而下运行；各节从 `Manuscript_Data.xlsx` 指定 worksheet 读列并出图。

若数据在 `data/` 子目录，notebook 中设置：

```python
from pathlib import Path
DATA_FILE = Path("data") / "Manuscript_Data.xlsx"
```

## 边界说明

- **可复现：** 论文 **定量图与统计分析**（基于 supplied workbook）。
- **不可复现：** 原始 EIM/超声/EMG 采集链路、电极可穿戴硬件、辅助机器人 **在线闭环** 控制器。
- **项目页：** 截至入库日 **未见** 独立 lab 项目页链到 GitHub；以 Zenodo 为唯一官方 artifact。

## 对 wiki 的映射

- 实体页 [`paper-bioimpedance-eim-wearable-myography.md`](../../wiki/entities/paper-bioimpedance-eim-wearable-myography.md) — 「工程实践 / 源码运行时序图」
