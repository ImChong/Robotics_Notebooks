---
type: source
source_type: paper
title: "FlyWire Nature 2024 全脑连接组与细胞类型谱系双篇主论文精读"
urls:
  - "https://doi.org/10.1038/s41586-024-07558-y"
  - "https://doi.org/10.1038/s41586-024-07686-5"
  - "https://codex.flywire.ai/"
created: 2026-11-14
updated: 2026-11-14
tags: [connectomics, drosophila, neuroscience, dataset, princeton, cambridge]
---

# FlyWire Nature 2024 全脑连接组与细胞类型谱系双篇主论文精读

- **阅读载体：** 核对主论文全文（Dorkenwald et al. 2024）、细胞分型主论文全文（Schlegel et al. 2024）与 Codex 门户在线说明摘录
- **核对范围：** 通读 Dorkenwald et al. (2024, `10.1038/s41586-024-07558-y`) Abstract / Main / Results / Discussion / Limitations / Methods / Data & Code availability，以及 Schlegel et al. (2024, `10.1038/s41586-024-07686-5`) Abstract / Main / Results / Discussion / Methods / Data & Code availability，并核对 Codex (`https://codex.flywire.ai/`) 2026 年公开数据集体量与访问限制
- **通读：** 已完成

## 一、主论文要点：Dorkenwald et al. (Nature 634, pp. 124–138, 2024-10-02)

- **DOI 与许可证：** `10.1038/s41586-024-07558-y`，CC-BY 4.0 Open Access。
- **成像基础与体积：** 成年雌性黑腹果蝇（*Drosophila melanogaster*）全脑连续切片透射电镜（ssTEM）数据 FAFB14.1，体素分辨率 `4 × 4 × 40 nm`，总神经毡（neuropil）体积 `0.0175 mm³`。
- **神经元规模（v783，139,255 个校对神经元）：**
  - **固有神经元（Intrinsic，118,501）：** 中央脑 32,388、视叶（optic lobes）77,536、视觉投射神经元（VPNs）8,053、视觉离心神经元（VCNs）524。
  - **传入神经元（Afferent）：** 感觉神经元 5,375（另统计口径 5,512 含细分）、上行神经元（ascending）2,362、复眼光感受器/固有传入 11,118、单眼（ocellar）273、eyelets 8。
  - **传出神经元（Efferent）：** 下行神经元（descending）1,303、运动神经元（motor）106、内分泌神经元（endocrine）80。
- **突触与递质规模：** 约 `2.44 × 10^8` 原始检测突触经滤波（去除冗余/低分阈值）后保留约 `1.30 × 10^8` 个化学突触连接，并附带突触前/后附着率（93.7% presynapses / 44.7% postsynapses 成功归属已校对骨干）及神经递质预测。
- **众包校对工作量与质量：** 累计投入约 `33.1 person-years`、完成 `3,013,513` 次交互编辑；在 826 个中央脑神经元基准评估中，按体积计 F1 达 `99.2%`。
- **已知成像与解剖局限：**
  - 左侧视板（left lamina）在制片阶段与髓质（medulla）发生部分机械分离伪影。
  - 原始电镜图像栈左右镜像反转；FlyWire 公开注释层已完成左右半球标签纠偏，但原始体数据（EM stack）本身仍呈镜像状态，跨数据集对齐时必须注意坐标系翻转。
- **静态归档与数据获取：** 除在线 CAVE / Codex（交互检索与服务端查询需 Google 账号登录）外，主发布快照通过 Zenodo 静态归档（`doi:10.5281/zenodo.10676866`、`doi:10.5281/zenodo.12588557`）。

## 二、细胞类型论文要点：Schlegel et al. (Nature 634, pp. 139–152, 2024-10-02)

- **DOI：** `10.1038/s41586-024-07686-5`。
- **层级分类与覆盖度：** 对全脑 139,255 个神经元汇总 `870,000+` 条注释，构建 `8,453` 个细胞类型，覆盖全脑 `96.4%` 的神经元（中央脑覆盖 `98%`，视叶覆盖 `92%`）。
- **谱系与形态分组：** 划分 `120` 个神经发育谱系（lineages）、`183` 个半谱系（hemilineages，占中央脑神经元约 `88%`）以及 `528` 个形态学组（morphology groups）。
- **跨脑对比与命名：**
  - 与 Janelia Hemibrain 匹配的共识细胞类型 `3,643` 个。
  - 新命名类型 `4,581` 个（其中 `3,504` 个采用 `CBXXXX` 编号体系）。
  - 保留文献经典类型 `229` 个。
- **跨个体连接组稳健性法则：**
  - 跨脑对比显示，突触数 `> 10` 或占下游总输入 `≥ 0.9%` 的连接超过 `90%` 在个体间保守；突触数 `> 31` 或 `> 2.6%` 的强连接保守率达 `99%`；边缘权重 `≤ 30%` 的差异往往源自技术重建噪声而非生物学个体差异。
  - 蘑菇体凯尼恩细胞（`KCg-m`）即使细胞数量在个体间存在 `30%–2x` 波动，系统通过降低平均输入类型数 `K`（如 `5.74 / 5.89` 对 `8.76` ALPN 类型）进行数量补偿，维持局部兴奋/抑制（E/I）平衡。
- **骨架与 NBLAST 归档：** 骨架化与跨脑 NBLAST 比对数据归档于 Zenodo（`doi:10.5281/zenodo.10877326`）。

## 三、Codex 门户与扩展数据集现状（2026-11-14 快照）

- Codex（`https://codex.flywire.ai/`）累计服务超 10 万登录用户（`103,172` 独立邮箱，`5.95M` 次请求，响应 `< 150 ms`）；交互检索与部分下载需 Google 登录认证。
- 门户同时托管 5 套连接组：
  - `FAFB v783 (CB)`：139,255 神经元 / 3,732,460 条去重边连接
  - `BANC v888`：158,262 神经元
  - `MANC v1.2.1`：23,665 神经元
  - `MAOL v1.1`：52,445 神经元
  - `MCNS v1.0`：166,700 神经元

## 参考链接

- [Whole-brain annotation and multi-connectome cell typing of Drosophila (Nature 2024)](https://doi.org/10.1038/s41586-024-07686-5)
- [Neuronal wiring diagram of an adult brain (Nature 2024)](https://doi.org/10.1038/s41586-024-07558-y)
- [FlyWire Codex](https://codex.flywire.ai/)
