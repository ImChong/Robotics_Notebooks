# Nordschleife Racer（yassin.app）

- **标题：** Nordschleife Racer — 在线可玩演示
- **类型：** site
- **链接：** https://yassin.app
- **入库日期：** 2026-08-23
- **一句话说明：** yassinsolim/nordschleife-racer 引擎的线上部署：单人计时、多人 Lobby、幽灵回放与全球排行榜（依赖 Supabase 后端）。

---

## 源码开放核查（2026-08-23）

| 项 | 结论 |
|----|------|
| **引擎代码** | **已开源** MIT → https://github.com/yassinsolim/nordschleife-racer |
| **可玩入口** | 站点 **Play Solo** / **Create Lobby** |
| **车体 3D 模型** | **未入库** — README 声明为第三方 mesh，由线上站点运行时加载 |
| **多人 / 排行榜** | 需作者部署的 **Supabase** 后端；本地仓为引擎模块，非独立可构建全站 |
| **测试** | 仓内 Playwright 回归针对线上构建 |

---

## 交叉链接

- 仓库：[`sources/repos/nordschleife_racer.md`](../repos/nordschleife_racer.md)
- Wiki：[`wiki/entities/nordschleife-racer.md`](../../wiki/entities/nordschleife-racer.md)
