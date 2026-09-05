# MILO 项目页（ac5113.github.io/MILO）

- **标题：** MILO: Reconstructing Humans and Objects in Interaction using Large Reconstruction Models
- **类型：** site
- **项目页：** <https://ac5113.github.io/MILO>
- **论文：** [arXiv:2608.27407](https://arxiv.org/abs/2608.27407)
- **代码：** <https://github.com/ac5113/MILO>
- **机构：** 德州大学奥斯汀分校（UT Austin）
- **入库日期：** 2026-08-31
- **再核日期：** 2026-09-05
- **一句话说明：** MILO 官方项目页：LRM 脚手架 + SMPL-H / 物体模板解释流程；InterCap / HODome / IMHD 结果与 BibTeX。

## 项目页要点（2026-09-05）

页首 TL;DR 三句，与论文摘要同构：

1. **LRMs as an HOI scaffold** — LRM 保住相对布局与邻近线索。
2. **Explain, don't fit** — 不要在歧义重投影里搜；解释 LRM 网格：人体拟合 SMPL-H，物体可选对齐模板。
3. **SOTA without contact** — InterCap / HODome / IMHD 用更弱信息（无 GT 接触）达到强精度。

方法段与论文 Fig. 2 一致：RGB → Hunyuan3D-2.0 联合网格 → 60 虚拟视角 → 关键点三角化 → SMPL-H + 多视角分割 → 可选模板。野外定性来自 PICO-db。致谢列 NSF-2504906 / 2544200 与 Adobe / Google / Nvidia 礼赠，以及 Hunyuan3D-2.0、HMR2.0、HaMeR、ViTPose、SMPL-H。

## 开源核查（2026-09-05 再核）

| 资产 | 状态 |
|------|------|
| 官方代码 | **已开源** <https://github.com/ac5113/MILO>（MIT；README 与项目页互指） |
| 论文 | [arXiv:2608.27407](https://arxiv.org/abs/2608.27407)（Accepted at ECCV 2026） |
| 权重 / 人体模型 | **不随仓**：SMPL-H / SMPL-X / VPoser / MANO 需注册；SAM 3 / SAM 3D 为 HF 门控 |
| 项目页 | 方法图、基准结果、BibTeX、致谢；页内 Code 链到 GitHub |

## 交叉链接

- 论文摘录：[milo_arxiv_2608_27407](../papers/milo_arxiv_2608_27407.md)
- 仓库：[ac5113/MILO](../repos/ac5113-milo.md)
- 主实体：[MILO](../../wiki/entities/paper-milo.md)
