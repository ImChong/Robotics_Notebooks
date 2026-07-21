# 天工造物开源社区（X-Humanoid Cloud）

> 来源归档

- **标题：** 天工造物开源社区
- **类型：** site（开源社区 / 文档 / 课程门户）
- **来源：** 北京人形机器人创新中心（X-Humanoid）
- **链接：** https://opensource.x-humanoid-cloud.com/
- **文档中心：** https://opensource.x-humanoid-cloud.com/plugin.php?id=zhanmishu_doc:index
- **课程中心：** https://opensource.x-humanoid-cloud.com/plugin.php?id=keke_video_base
- **开源项目专题：** https://opensource.x-humanoid-cloud.com/portal.php?mod=topic&topicid=1
- **GitHub 组织（社区页指向）：** https://github.com/Open-X-Humanoid
- **入库日期：** 2026-07-21
- **一句话说明：** Discuz 风格的「天工造物」开源社区：聚合博客、问答、开源项目索引、结构化文档与视频课程；是官网之外面向开发者的日常跟踪入口。
- **开源状态：** **已开源（文档 + 链出代码）** — 文档中心托管天工 Lite/Pro SDK/手册；首页生态索引链到 GitHub / 项目页；直接 HTTP 抓取可能遇反爬，正文以 Jina Reader 为准。
- **沉淀到 wiki：** 是 → [`wiki/entities/x-humanoid.md`](../../wiki/entities/x-humanoid.md)

---

## 为什么值得保留

- **开发者运营面**：问答（灵巧手 / 运动控制 / 数据集 / 本体）与官方公告帖（XR-1、RoboMIND、ArtVIP、TienKung-Lab、训练工具链）集中在此，比官网新闻页更贴近工程问题。
- **文档中心即 Lite/Pro 手册入口**：与 `TienKung_Docs` GitHub 镜像互补；社区文档中心声明「浏览文档」并链 [GitHub TienKung_Docs](https://github.com/x-humanoid-robomind/TienKung_Docs)（历史 org）/ 现行 [`Open-X-Humanoid/TienKung_Docs`](https://github.com/Open-X-Humanoid/TienKung_Docs)。
- **课程与 Sim2Real**：`TienKung-Lab` README 横幅链到课程中心 `keke_video_base`（course cid=27），是仿真到真机培训入口。

## 首页生态索引（2026-07-21）

社区首页「开源生态索引」侧栏 / 热链（原文链接）：

| 名称 | URL |
|------|-----|
| 天工机器人本体 | https://opensource.x-humanoid-cloud.com/portal.php?mod=topic&topicid=1 |
| RoboMIND 数据集 | https://x-humanoid-robomind.github.io/ |
| 训练工具链 | https://github.com/Open-X-Humanoid/x-humanoid-training-toolchain |
| 运动控制框架 | https://github.com/Open-X-Humanoid/TienKung-Lab |
| 数字资产 ArtVIP | https://x-humanoid-artvip.github.io/ |
| 具身世界模型 WoW | https://wow-world-model.github.io/ |
| 多模态大模型 Pelican-VL | https://pelican-vl.github.io/ |
| XR-1 VLA 模型 | https://github.com/Open-X-Humanoid/XR-1 |
| BicMap（门户推广） | https://bicmap.x-humanoid-cloud.com/ |

## 文档中心摘录（`zhanmishu_doc:index`）

标题：**天工造物开源社区文档中心**。

| 文档 | 要点（社区页） |
|------|----------------|
| 天工 **Lite** SDK 开发文档 | 纯电驱拟人奔跑全尺寸；约 **20 DoF**；约 **6 km/h** 奔跑；避障/上下坡/抗冲击 |
| 天工 **Pro** SDK 开发文档 | 头 3 DoF；单臂 7 DoF（肩肘腕）；上下半身整机 |
| 天工 **Pro** 用户手册 | 最多约 **42** 全身 DoF；母平台定位 |
| 天工 **Lite** 用户手册 | 单臂 4 DoF、单腿 6 DoF；整机 **20** 关节电机 |

导航模块：博客、问答、开源项目、文档、课程中心、社区（专家介绍）。

## 抓取与访问注意

- 根路径与部分插件页对裸 `curl` 返回反爬脚本（`nox` / `gangplank`）；**Agent 复抓优先用可读代理或浏览器**。
- 论坛帖 URL、活动页会随运营变化；wiki 只固化**稳定入口**（文档中心、生态索引表、GitHub）。

## 与本仓库现有资料的关系

- 官网归档：[`sources/sites/x-humanoid.md`](./x-humanoid.md)
- 组织仓归档：[`sources/repos/open-x-humanoid.md`](../repos/open-x-humanoid.md)
- 微信策展入口：[`sources/blogs/wechat_jixie_robot_open_source_treasury_issue01_10_robots.md`](../blogs/wechat_jixie_robot_open_source_treasury_issue01_10_robots.md)

## 对 wiki 的映射

- 升格 [`wiki/entities/x-humanoid.md`](../../wiki/entities/x-humanoid.md)「社区与文档入口」节。
- 交叉更新 [`wiki/entities/tienkung-humanoid-open-source.md`](../../wiki/entities/tienkung-humanoid-open-source.md)。
