# LeTools 官方文档（docs.html）

> 来源归档（ingest）

- **类型：** 文档站（静态 SPA + Markdown）
- **Learning 入口：** <https://www.letools.lejurobot.com/docs.html?type=learning>
- **Skills 入口：** <https://www.letools.lejurobot.com/docs.html?type=skills>
- **默认 type：** `learning`（未带查询参数时）
- **入库日期：** 2026-08-17
- **抓取说明：** 文档引擎从 `${docDir}/menu.json` 与 `${docDir}/${file}` 拉取；`type=skills` 时 `docDir=skills_docs`，否则 `docs`。无头抓取若只拿 HTML 骨架会报 “Error loading docs”，属前端 `fetch` 失败而非站点下线。

## 一句话

官方文档把 **LeTools-Learning（训策略）** 与 **LeTools-Skills（原子技能/行为树）** 分成两套目录，用同一 `docs.html` 切换。

## 开源状态（步骤 2.5）

- Markdown 与 `menu.json` **公开可抓**；实现细节仍以 GitHub 仓为准。
- Skills 文档侧栏截至入库日 **明显薄于** Learning（仅 Beginner + FAQ），深度内容在 [letools_opensource README](https://github.com/LejuRobotics/letools_opensource) 与 `skills/README.md`。

## Learning 目录（`docs/menu.json`）

| 分组 | 页面 |
|------|------|
| GET STARTED | LeTools Learning · Beginner's Introduction · Installation |
| TUTORIALS | Quick Start · Data Preparation · Model Training · Lerobot Training · Inference · Bring Your Own Policies |
| Troubleshooting | FAQ · Community |

对应哈希示例：`docs.html#tutorials/data_preparation.md`（Learning README 亦链到这些锚点）。

## Skills 目录（`skills_docs/menu.json`）

| 分组 | 页面 |
|------|------|
| GET STARTED | Beginner's Introduction |
| TROUBLESHOOTING | FAQ |

## 与仓库 README 的分工

| 读者目标 | 先看 |
|----------|------|
| rosbag → LeRobot → ACT/π/GR00T/LingbotVLA → sim/真机 | Learning 文档 + [LeTools-Learning](https://github.com/LejuRobotics/LeTools-Learning) |
| 行为树 JSON、SkillBase、SDK 直调、dry-run | Skills 文档 + [letools_opensource](https://github.com/LejuRobotics/letools_opensource) / `skills/README.md` |
| 安装报错、QQ/微信群、企业合作 | Learning Troubleshooting / Community |

## 对 wiki 的映射

- 升格：[wiki/entities/letools.md](../../wiki/entities/letools.md)
- 产品站：[letools-lejurobot.md](letools-lejurobot.md)
