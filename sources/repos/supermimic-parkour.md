# SuperMimic Parkour

> 来源归档

- **类型：** repo（受 SuperMimic 启发的社区实现）
- **链接：** <https://github.com/yhx1203/supermimic_parkour>
- **核查版本：** [f398519](https://github.com/yhx1203/supermimic_parkour/tree/f398519453a25b3f10722626b7356266d8b7f875)
- **数据：** [PHP Parkour motion_matching](https://github.com/amazon-far/php_parkour/tree/main/motion_matching)，动作数据不随本仓库提供。
- **许可证：** Apache-2.0；第三方数据的条款需另行核查。
- **核查日期：** 2026-11-01
- **沉淀到 wiki：** [SuperMimic Parkour](../../wiki/entities/supermimic-parkour.md)

## 内容与核查范围

README 提供安装、五类 PHP 动作数据组织方式、训练及浏览器回放入口。作者报告在单 GPU 上训练略超四小时得到越过 130 cm 障碍的策略；未提供足够条件证明可复现或真机有效。

已阅读 README、训练与回放入口、launcher、训练循环及数据加载相关代码；未完成所有必要源码的逐项核验，也未运行训练。流程图和时序图是已核查主路径，后续应复核奖励、状态、地形及评测实现。
