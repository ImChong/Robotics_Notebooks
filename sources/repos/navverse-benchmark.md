# NavVerse-Benchmark

> 来源归档

- **标题：** NavVerse-Benchmark — A Physically Grounded Indoor-to-Outdoor Embodied Navigation Benchmark
- **类型：** repo（当前以项目站托管为主）
- **机构：** 密歇根大学（UMich）；CURLY 实验室（UMich-CURLY）
- **链接：** <https://github.com/UMich-CURLY/NavVerse-Benchmark>
- **项目页：** <https://umich-curly.github.io/NavVerse-Benchmark/>
- **论文：** <https://arxiv.org/abs/2607.19695>
- **默认分支：** `website`（GitHub Pages 静态站）
- **入库日期：** 2026-09-13
- **许可证：** 仓内未声明 LICENSE 文件（截至核查日）
- **代码 / 开源状态：** **待发布** — 仓已公开但 **仅含项目站静态资源**；Isaac Sim 仿真与评测入口标注 Coming soon
- **一句话说明：** NavVerse 官方 GitHub：当前托管项目页资产；benchmark 代码与数据待官方发布。
- **沉淀到 wiki：** 是 → [`wiki/entities/paper-navverse.md`](../../wiki/entities/paper-navverse.md)
- **交叉归档：** [navverse-benchmark-github-io.md](../sites/navverse-benchmark-github-io.md)、[navverse_arxiv_2607_19695.md](../papers/navverse_arxiv_2607_19695.md)

---

## 仓内结构（2026-09-13 快照，`website` 分支）

| 路径 | 作用 |
|------|------|
| `index.html` | 项目站入口 |
| `bundle.js` / `bundle.css` | 前端 bundle |
| `corl_figs/` | 论文/榜单配图 |
| `web-media/` | rollout 视频与示意图 |
| `tools/` | 站点构建/媒体工具（非 benchmark runner） |

**注意：** `main` 分支不存在；`git clone` 后默认检出 `website`，**无** Python 评测脚本或 Isaac Sim 环境注册。

---

## 对 wiki 的映射

- 实体页：[NavVerse](../../wiki/entities/paper-navverse.md)
- 任务交叉：[VLN](../../wiki/tasks/vision-language-navigation.md)、[零样本物体导航](../../wiki/tasks/zero-shot-object-navigation.md)
- 基准对照：[VLN-CE](../../wiki/entities/paper-vln-02-vln-ce.md)、[ESI-Bench](../../wiki/entities/esi-bench.md)
