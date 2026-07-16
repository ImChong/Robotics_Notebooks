# regrind

> 来源归档（ingest）

- **标题：** yunhaif/regrind
- **类型：** repo
- **原始链接：** <https://github.com/yunhaif/regrind>
- **许可证：** MIT
- **入库日期：** 2026-07-16
- **一句话说明：** REGRIND 官方实现：含预计算重定向轨迹、Drake 重定向脚本、RL 训练与部署；`pydrake`/MOSEK 为可选重定向依赖。

## 核心摘录

- **重定向入口：** `python scripts/retarget_hand_object.py --robot {leaphand,wujihand} --object {scissors,screwdriver}`；需 `source scripts/set_path.sh`。
- **可跳过重定向：** 仓库内已附带 retargeted trajectories，可直接 RL training。
- **依赖：** 核心 `import regrind` 不强制 Drake；重定向需单独安装 Drake + MOSEK license。
- **对 wiki 的映射：** 见 [REGRIND（重定向引导灵巧操作 RL）](../../wiki/methods/regrind-retargeting-guided-rl.md)
