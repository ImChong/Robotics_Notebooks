# RoboParty 人形机器人运动控制 Know-How

> 来源归档（技术知识库；本次核查目录与部分正文）

- **URL：** <https://roboparty.feishu.cn/wiki/GvUxwKVeNiGa7kku6vEcvqfKn87>
- **机构：** 上海萝博派对科技有限公司（RoboParty）
- **类型：** site（持续维护的技术教程）
- **入库 / 核查日期：** 2026-10-06
- **官方索引：** [Party OS README](https://github.com/Roboparty/Party_OS) → [来源归档](../repos/party_os.md)
- **代码：** <https://github.com/Roboparty>（正文指向官方组织；教程示例偏伪代码，可运行项目见各独立仓）
- **相关文档：** [Roboto Origin 完整研发文档](roboparty_com_roboto_origin_doc.md)
- **读取方式：** 按 [Agent Reach](https://github.com/Panniantong/agent-reach) 技能使用 Jina Reader，得到开篇与传统控制学习路线的部分正文；Chrome DevTools MCP 读取公开页目录并抽查正文。未保存全文。
- **日期边界：** 页面显示「Last updated: Sep 03 / Modified September 3」，未显示年份；不推定为完整版本发布日期。缺少历史目录快照，不能判断哪些章节最近新增。

## 已核查的组织方式

文档先给学习路线与控制问题的建模思路，再按传统模型控制和学习控制展开。作者说明每种方法依次介绍原理、示例与局限，并提醒示例代码偏伪代码，完整复现要查官方开源项目。

| 目录主题 | 可复用的阅读坐标 |
| --- | --- |
| 趋势与学习路线 | MPC → BFM 的范式变化；硬件、数据、算法与评测问题分开 |
| 传统控制 | OCP、LIP/ZMP、SLIP/VMC、WBD/WBC/TSID、SRBD/凸 MPC、质心模型/NMPC、状态估计 |
| 学习控制 | RL、Teacher–Student/DAgger、DreamWaq、PIE、Attention 落足点、Retarget、DeepMimic |
| 工程学习顺序 | URDF 与 Pinocchio 运动学/浮动基动力学 → MuJoCo → QP 求解器实践 → MPC/WBC |

## 本次核查边界

- 已确认目录与开篇组织方式，未逐章深读；章节有目录不代表代码或公式完整。
- 本次只服务公司路线导航；后续逐章 ingest 先复用已有控制、重定向与模仿学习页，再判断缺口。
- 微信公众号重试触发验证码，且当前环境未安装 Camoufox 抓取链；已有 [Lab 成立原文归档](../blogs/wechat_roboparty_lab_party_os_3_tools.md)继续作为历史来源，不声称本次重抓成功。

## 对 wiki 的映射

- [RoboParty 公司](../../wiki/entities/roboparty.md)
- [Party OS](../../wiki/entities/party-os.md)
- [RoboParty Lab / Party OS 技术地图](../../wiki/overview/roboparty-lab-party-os-technology-map.md)
