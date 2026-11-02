# [具身开源]从开源仓库到真机差几步：人形机器人复现门槛与 Sim2Real 缺口

> 来源归档（blog / 微信公众号）

- **标题：** [具身开源]从开源仓库到真机差几步：人形机器人复现门槛与 Sim2Real 缺口
- **类型：** blog
- **作者：** 具身智能DAI / 智生于形（微信公众号）
- **原始链接：** https://mp.weixin.qq.com/s/94TLMeN9fgAIt68jQ6dgMA
- **发表日期：** 2026-11-02
- **入库日期：** 2026-11-02
- **抓取方式：** `scripts/fetch_wechat_article.py`（Chromium + 微信内置 UA），正文 4309 字符
- **一句话说明：** 把开源人形仓库复现的难点拆成驱动与实时、标定、环境依赖三道坎，叠加 Sim2Real 缺口，给出工时量级与四项门禁清单。
- **说明：** 这是工程经验综述，**没有论文或具体开源项目清单**，所以不做逐项目节点；知识沉淀到 [open-source-humanoid-reproduction-gate](../../wiki/concepts/open-source-humanoid-reproduction-gate.md)。文中工时、倍数、经验数值均为作者表述（文章自称「经验示意，不是任何项目公开报价」），未核实。文中提到的「E2 七步核查流程」是作者另一篇文章，本库未收录。

## 文章结构

1. 开篇五点摘要；Sim2Real 缺口拆为动力学模型、执行器响应、传感器噪声与延迟、控制频率与通信延迟四个层面
2. 第一道坎：驱动与实时（EtherCAT / CAN-FD、实时内核、分层实时、过热 / 供电 / IMU 温漂 / 线缆）
3. 第二道坎：标定文件与零位（零位、方向、URDF 对齐、力矩补偿、手眼标定 AX=XB）
4. 第三道坎：环境与依赖（容器、锁版本）
5. 三道坎的工时量级与成本结构
6. 复现门禁清单（四关）
7. 把门槛写进预算的四条建议

## 对 wiki 的映射

- 新页：[open-source-humanoid-reproduction-gate](../../wiki/concepts/open-source-humanoid-reproduction-gate.md)
- 回链：[开源人形硬件对比](../../wiki/entities/open-source-humanoid-hardware.md)
