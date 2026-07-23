# Evo-SOTA.io（MINT-SJTU/Evo-SOTA.io）

> 来源归档

- **标题：** VLA & Dexterous Manipulation SOTA Leaderboard
- **类型：** repo
- **组织：** MINT-SJTU / EvoMind Tech
- **代码：** <https://github.com/MINT-SJTU/Evo-SOTA.io>
- **线上站：** <https://sota.evomind-tech.com/>
- **许可：** MIT
- **入库日期：** 2026-07-23
- **一句话说明：** VLA / 灵巧手多基准排行榜的 **Next.js 静态站源码**；论文摘录分数存于 `public/data/*.json`，支持开源过滤、RL/SFT 分类与中英双语。
- **沉淀到 wiki：** [VLA SOTA Leaderboard](../../wiki/entities/vla-sota-leaderboard.md)

## 与本仓库知识的关系

| 主题 | 关系 |
|------|------|
| [VLA](../../wiki/methods/vla.md) | 桌面操作 VLA 相对位次导航入口 |
| [Evo-1](../../wiki/entities/paper-evo1-lightweight-vla.md) | 同生态维护方；Meta-World 等榜条目 |
| [FabriVLA](../../wiki/entities/paper-fabrivla.md) | Meta-World 榜开源区前列条目（入库日） |
| [具身评测选型闭环](../../wiki/queries/embodied-eval-benchmark-selection-loop.md) | 第 ③ 层「策略成功率」的**社区聚合视图**（非官方基准本体） |

## 工程要点（README 摘要）

- **本地：** Node 18+；`npm install && npm run dev` → `localhost:3000`；`npm run build` 输出 `out/`。
- **贡献：** Issue / PR 提交论文链接与分数；联系 `business@evomind-tech.com`。
- **数据注意：** 结果来自原文或第三方复现，可能因协议差异而变。

## 为何值得保留

- 榜站可本地重建；JSON 数据便于核对站内 wiki 数字是否过时。
