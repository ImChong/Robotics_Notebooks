# NavVerse 项目页（umich-curly.github.io/NavVerse-Benchmark）

> 来源归档

- **标题：** NavVerse — Benchmarking Indoor-to-Outdoor Embodied Navigation
- **类型：** site / project-page / benchmark
- **URL：** <https://umich-curly.github.io/NavVerse-Benchmark/>
- **论文：** <https://arxiv.org/abs/2607.19695>
- **GitHub：** <https://github.com/UMich-CURLY/NavVerse-Benchmark>
- **机构：** 密歇根大学（UMich）；CURLY 实验室
- **入库日期：** 2026-09-13
- **一句话说明：** 官方项目站：场景/任务/指标说明、零样本基线榜单、室内–户外 transition 失败诊断与 rollout 定性示例。

## 开源核查（步骤 2.5，截至 2026-09-13）

| 核查项 | 结论 |
|--------|------|
| 项目页是否链到 arXiv | 页内标注 **Paper Coming soon**（arXiv 已上线，站点链接待补） |
| 项目页是否链到代码 | 标注 **Code Coming soon** |
| GitHub 仓内容 | 仅 `website` 分支静态站（`index.html` / `bundle.js` / 媒体资源） |
| 可运行 Isaac Sim 评测 | **未见** |
| 综合判定 | **待发布** — 项目页与 arXiv 可用；benchmark 代码与数据待发布 |

## 公开信息要点

- **场景：** 100 indoor / 50 urban outdoor / 50 indoor-to-outdoor；城市 enrichment + POI 店面 + 门–立面 hybrid assembly。
- **任务：** ObjNav / PlaceNav / VLN；PlaceNav 要求语义地点（餐厅、咖啡馆、银行等）接地。
- **指标：** SR、SPL、CE、CR、ADO、NSR；强调「只看 SR 会隐藏物理失败」。
- **基线（零样本）：** SGImagineNav（模块化）、PoliFormer（RL）、UniNaVid（VLA）、LongNav-R1（VLA-RL）。
- **主要读点：** UniNaVid 任务完成最高；SGImagineNav 安全指标最好；PlaceNav outdoor→indoor-to-outdoor 跌幅最大；许多 transition episode 在到达户外前就失败。

## 关联资料

- 论文摘录：[`sources/papers/navverse_arxiv_2607_19695.md`](../papers/navverse_arxiv_2607_19695.md)
- 仓库归档：[`sources/repos/navverse-benchmark.md`](../repos/navverse-benchmark.md)
- Wiki 实体：[`wiki/entities/paper-navverse.md`](../../wiki/entities/paper-navverse.md)
