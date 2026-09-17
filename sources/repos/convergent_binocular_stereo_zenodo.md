# Convergent Binocular Stereo（Zenodo 代码 + CBS-BM 数据集）

> 来源归档

- **标题：** Convergent Binocular Stereo: Depth Perception for Humanoid Robot Vision
- **类型：** repo / Zenodo artifact / dataset
- **链接：** <https://zenodo.org/records/21053380>
- **DOI：** <https://doi.org/10.5281/zenodo.21053380>
- **许可证：** MIT
- **发布：** 2026-07-17
- **论文：** [Science Robotics eaec7205](../papers/convergent_binocular_stereo_scirobotics_2026.md)
- **关联标定：** [dijit-binocular-robotic-head](dijit-binocular-robotic-head.md)（GitLab）
- **入库日期：** 2026-09-17
- **一句话说明：** CBS 会聚立体 **官方复现包**：`cbs-convergent-binocular-stereo.zip`（~2 GB，代码 + CBS-BM）与 `plots.zip`（~17 GB，手稿图表复现）；入口 `python3 run_one.py`，核心算法 `algo.py`。
- **开源状态：** **已开源**（Zenodo，无 GitHub 持续维护仓）

---

## 包内容与入口

|  artifact | 说明 |
|-----------|------|
| `cbs-convergent-binocular-stereo.zip` | CBS 算法 + CBS-BM 数据集 + `run_one.py` |
| `plots.zip` | `histogram_fig6.py`、`histogram_sup.py`；`plots/analysis/` 表格复现 |
| `CBS-BM/CBS-BM.zip` | 数据集（需解压） |

### 示例运行（Zenodo README）

```bash
# 解压数据集后，示例场景 5、fixation 3
python3 run_one.py --scene 5 --fixation 3 \
  --L_parallel_motors "67,116" --R_parallel_motors "88,101" \
  --dataset "CBS-BM/CBS-BM" \
  --calib_file "CBS-BM/CBS-BM/calib.yml" --b 0.115 \
  --motors_calib_L "CBS-BM/CBS-BM/motors/L" \
  --motors_calib_R "CBS-BM/CBS-BM/motors/L"
```

自定义图像对：传入 `--fundamental`、`--im_L`、`--im_R` 及标定路径（见 Zenodo 页完整参数）。

---

## CBS-BM 目录结构（摘要）

- **49 场景**（索引 0–48）：每场景含 `fixations.csv`、`gt_disp.npy`、平行 rectified 对图、多 fixation 子目录 `fixations/<id>/{L,R}.png`
- **`motors/L|R/`**：电机标定文件
- **`calib.yml`**：相机内参（原分辨率；图像为 1/2 缩放时需调整焦距/主点）

---

## 核心模块

| 路径 | 作用 |
|------|------|
| `run_one.py` | CLI：加载场景/fixation 或自定义图像对，调用 CBS |
| `algo.py` | Gabor + 粗到细视差估计（内存密集） |
| `plots/` | 手稿统计图与表格复现 |
| `CBS-BM/` | 基准数据与标定 |

---

## 可复现边界

| 可做 | 不可做 / 未包含 |
|------|----------------|
| 在 CBS-BM 上复现论文 disparity/depth 对比 | 一键 DIJIT 真机闭环控制 |
| 自定义会聚图像对 + 已知 F 矩阵评测 | GitHub Issues / CI |
| 复现 Fig.6 直方图与表格（plots 包） | 完整 17 GB plots 需大磁盘 |

---

## 交叉链接

- 论文归档：[convergent_binocular_stereo_scirobotics_2026.md](../papers/convergent_binocular_stereo_scirobotics_2026.md)
- Wiki 实体：[paper-convergent-binocular-stereo.md](../../wiki/entities/paper-convergent-binocular-stereo.md)
- DIJIT GitLab：[dijit-binocular-robotic-head.md](dijit-binocular-robotic-head.md)
