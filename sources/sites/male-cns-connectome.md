# Male CNS Connectome

> 来源归档

- **标题：** Male CNS Connectome（果蝇雄性中枢神经系统连接组）
- **类型：** site
- **机构：** HHMI Janelia / FlyEM Project Team
- **链接：** https://male-cns.janelia.org/
- **下载页：** https://male-cns.janelia.org/download/
- **Janelia 项目页：** https://www.janelia.org/project-team/flyem/male-cns-connectome
- **论文：** Berg et al., *Sexual dimorphism in the complete connectome of the Drosophila male central nervous system* — [Cell 2026](https://www.cell.com/cell/fulltext/S0092-8674(26)00942-6)（bioRxiv [2025.10.09.680999](https://www.biorxiv.org/content/10.1101/2025.10.09.680999v2)）
- **数据集 ID：** `male-cns:v1.0`
- **许可：** CC-BY
- **入库日期：** 2026-09-10
- **一句话说明：** 首个完整 proofread 的雄性果蝇 CNS（中枢脑 + 视叶 + 腹神经索）突触分辨率连接组，可与雌性 FlyWire 连接组跨性别比较。
- **代码/工具：** 数据开放下载；探索依赖 [neuPrint](https://neuprint.janelia.org/)、[Neuroglancer](https://github.com/google/neuroglancer)、[neuprint-python](https://github.com/connectome-neuprint/neuprint-python) 等生态（见各 `sources/` 归档）
- **沉淀到 wiki：** 是 → [`wiki/entities/male-cns-connectome.md`](../../wiki/entities/male-cns-connectome.md)、[`wiki/concepts/fly-connectomics-stack.md`](../../wiki/concepts/fly-connectomics-stack.md)

---

## 核心定位

- **范围：** 覆盖果蝇 **中枢脑、视叶（类比哺乳动物视网膜）与腹神经索 VNC（类比脊髓）**，含完整 **颈部连接**。
- **里程碑：** 首个 **雄性** 全 CNS 连接组；与既有 **雌性** 连接组（FlyWire 等）构成首个可跨性别突触分辨率比较资源。
- **版本：** **v1.0** 于 2026-06-08 发布；论文 2026-09-03 发表于 *Cell*。

---

## 科学要点（Janelia 项目页 / 论文）

| 维度 | 要点 |
|------|------|
| 性别特异性 | **262** 种 sex-specific + **114** 种 sexually dimorphic 细胞类型，占中枢脑 **4.8%** |
| 结构分布 | 性别特异/二态神经元集中于 **高阶脑区**；感觉/运动外周 largely isomorphic |
| 连接传播 | 少量二态神经元经 **dimorphic connectivity** 向全脑传播影响 |
| 应用 | 跨 CNS 回路探索、视觉-运动通路、性行为相关回路比较 |

---

## 数据与访问方式

### 交互探索

- [neuPrint](https://neuprint.janelia.org/?dataset=male-cns%3Av1.0) — 连接查询
- [Male CNS Cell Type Explorer](https://reiserlab.github.io/celltype-explorer-drosophila-male-cns) — 细胞类型与 eyemap
- [Clio](https://clio.janelia.org/) — 注释导向浏览
- [Neuroglancer 场景](https://neuroglancer-demo.appspot.com/#!gs://flyem-male-cns/v1.0/male-cns-v1.0.json) — 体数据 + 分割 + 突触 + 神经纤维网室
- [Dimorphism Explorer](https://male-cns.janelia.org/build/dimorphism_overview/) — 雌雄二态比较

### 程序化访问

- **Python：** `neuprint-python`，`dataset='male-cns:v1.0'`，需 neuPrint 账号 API token
- **R：** `neuprintr` / `malecns`（natverse）
- **形态学：** [navis](https://github.com/navis-org/navis) + `navis-flybrains` 做坐标变换

### 原始数据下载（download 页）

| 类别 | 示例 | 格式/规模 |
|------|------|-----------|
| 体数据 EM | `gs://flyem-male-cns/em/em-clahe-jpeg` | precomputed，8 nm 各向同性 |
| 分割 | `gs://flyem-male-cns/v1.0/segmentation` | proofread v1.0 |
| 注释 | `body-annotations-male-cns-v1.0-minconf-0.5.feather` | Apache Feather，~13 MB |
| 连接图 | `connectome-weights-male-cns-v1.0-minconf-0.5.feather` | 全图 segment-to-segment，~1.1 GB |
| 突触点 | `syn-points-*.feather` | ~12.7 GB |
| 骨架 | 多格式/坐标空间可下载 | 见 download 页 Skeletons 节 |

读取示例：`cloud-volume` / `tensorstore` 读 precomputed；`pyarrow` 读 Feather 表。

---

## 开源核查（步骤 2.5）

| 项 | 结论 |
|----|------|
| 数据集 | **已开放**（CC-BY）；体数据、注释、连接表、骨架可下载 |
| 探索服务 | neuPrint / Neuroglancer 嵌入为 **在线服务**，非本仓库代码 |
| 官方代码仓 | 无单一「MaleCNS 训练代码」仓；工具链分散于 Neuroglancer、neuprint-python、DVID 等 |

---

## 对 wiki 的映射

- [Male CNS Connectome](../../wiki/entities/male-cns-connectome.md)
- [果蝇连接组工具栈](../../wiki/concepts/fly-connectomics-stack.md)
- [FlyWire](../../wiki/entities/flywire.md) — 雌性全脑连接组对照
- [neuPrint](../../wiki/entities/neuprint.md)
- [Neuroglancer](../../wiki/entities/neuroglancer.md)
