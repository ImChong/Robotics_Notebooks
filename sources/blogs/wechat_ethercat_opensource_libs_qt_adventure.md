# EtherCAT 开源库大全（性能微秒级别）

> 来源归档（blog / 微信公众号）

- **标题：** EtherCAT 开源库大全（性能微秒级别）
- **类型：** blog / curated-index
- **作者：** Qt历险记（微信公众号）
- **原始链接：** https://mp.weixin.qq.com/s/C34QLsVb7Sd-GlvPvQlGKQ
- **入库日期：** 2026-10-14
- **抓取方式：** WebFetch 被微信「环境异常」验证页拦截；改用 curl + 微信客户端 User-Agent 取得 HTML，提取 `js_content` 正文
- **原始抓取落盘：** [`sources/raw/wechat_ethercat_opensource_libs_qt_adventure_2026-10-14.md`](../raw/wechat_ethercat_opensource_libs_qt_adventure_2026-10-14.md)
- **一句话说明：** EtherCAT 主站 / 从站 / 语言绑定 / CiA402 伺服 / 仿真五类开源库清单，附 GitHub 地址与选型一句话。
- **沉淀到 wiki：** [`wiki/overview/ethercat-opensource-libs-wechat.md`](../../wiki/overview/ethercat-opensource-libs-wechat.md)

## 步骤 2.5：开源核查（2026-10-14）

本次会话的 GitHub 访问仅限本仓库，**无法打开下表仓库**，故「文章声明」列只是文章原话，**未独立核查**；未给 URL 的项不臆造地址。

| # | 文内名称 | 文章给出的地址 | 文章声明 / 本次结论 | wiki 节点 |
|---|----------|----------------|---------------------|-----------|
| 1 | SOEM | github.com/OpenEtherCATsociety/SOEM | 已有归档 [sources/repos/soem.md](../repos/soem.md)（2026-08-12 核查） | [soem](../../wiki/entities/soem.md) |
| 2 | IgH EtherCAT Master | gitlab.com/etherlab.org/ethercat | 声明公开（GitLab），未核查 | [igh-ethercat-master](../../wiki/entities/igh-ethercat-master.md) |
| 3 | KickCAT | github.com/leducp/KickCAT | 声明公开，未核查 | [kickcat](../../wiki/entities/kickcat.md) |
| 4 | Ethercrab | github.com/ethercrab-rs/ethercrab | 声明公开，未核查 | [ethercrab](../../wiki/entities/ethercrab.md) |
| 5 | CherryECAT | 「RT-Thread 社区」（无 URL） | 未能确认仓库 | [cherryecat](../../wiki/entities/cherryecat.md) |
| 6 | libethercat（DLR） | 无 | 未能确认仓库 | [libethercat-dlr](../../wiki/entities/libethercat-dlr.md) |
| 7 | SOES | github.com/OpenEtherCATsociety/SOES | 声明公开，未核查 | [soes](../../wiki/entities/soes.md) |
| 8 | Beckhoff SSC | 无（ETG 官网申请） | 受限分发，非公开仓库 | [beckhoff-ssc](../../wiki/entities/beckhoff-ssc.md) |
| 9 | pysoem | 无 | 未能确认仓库 | [pysoem](../../wiki/entities/pysoem.md) |
| 10 | soem-qt | 无 | 未能确认仓库（可能为泛称） | [soem-qt](../../wiki/entities/soem-qt.md) |
| 11 | ROS2-ethercat | 无 | 未能确认仓库（可能为泛称） | [ros2-ethercat](../../wiki/entities/ros2-ethercat.md) |
| 12 | ecat_servo | github.com/kubabuda/ecat_servo | 声明公开，未核查 | [ecat-servo](../../wiki/entities/ecat-servo.md) |
| 13 | OpenSML | github.com/feecat/OpenSML | 声明公开，未核查 | [opensml](../../wiki/entities/opensml.md) |
| 14–15 | EtherCAT Master Sim / ethercat-master-sim | 无 | 两条重合，合并为一个节点；未能确认仓库 | [ethercat-virtual-slave-sim](../../wiki/entities/ethercat-virtual-slave-sim.md) |

## 对 wiki 的映射

- [ethercat-opensource-libs-wechat](../../wiki/overview/ethercat-opensource-libs-wechat.md) — 索引与选型
