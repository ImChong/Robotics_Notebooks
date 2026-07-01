# HRDexDB 项目页（snuvclab.github.io/HRDexDB）

- **标题**: HRDexDB: A Paired Human-Robot Dataset for Cross-Embodiment Dexterous Grasping
- **类型**: dataset / research-portal
- **项目页**: <https://snuvclab.github.io/HRDexDB/>
- **论文**: Lim et al., arXiv:[2604.14944](https://arxiv.org/abs/2604.14944)（2026）
- **机构**: Seoul National University · RLWRLD
- **收录日期**: 2026-07-01

## 一句话摘要

首个在**相同物体**上配对采集人类与多种灵巧机器人手抓取序列的大规模数据集：**2.1K** sequences · **100+** objects · **5** embodiments · **23** 路同步相机 · 精确 **3D** 手/机/物标注 · Inspire / Allegro **触觉**流。

## 为何值得保留

- **跨 embodiment 配对缺口**：相对仅人类 HOI（DexYCB、GigaHands）或仅机器人 teleop（RealDex）的数据集，强调**同物体、可比抓取动作**下的人–机对齐。
- **多模态完备性**：多视角 RGB + ego + 运动学 + 物体 6D + 触觉 + 接触可视化，且 **markerless** 不牺牲 RGB 质量。
- **多种灵巧手形态**：覆盖 Allegro 与两款 Inspire，便于研究 morphology 对抓取策略与迁移的影响。

## 公开要点（编译自项目页）

| 字段 | 内容 |
|------|------|
| 序列 | **2.1K** grasping trials |
| 物体 | **100+** diverse objects |
| Embodiment | **5**（人 + 多种灵巧机器人手） |
| 相机 | **23** 路全同步（密集第三人称 + ego） |
| 标注 | 手/机器人/物体 **3D** 时空轨迹；接触区域可视化 |
| 触觉 | **Inspire & Allegro** 系列触觉信号（相对其他机器人数据集更完整） |
| 发布 | 论文标注将公开完整数据集（投稿时 100 物体，规划扩至 1000） |

## 对 Wiki 的映射

- **wiki/entities/hrdexdb-dataset.md**：数据集实体页
- **sources/papers/hrdexdb_arxiv_2604_14944.md**：论文级技术细节与 Table 1 对照
