# 自变量机器人官网 Blog / Research 列表核查

- **类型：** 官方站点索引（技术博文与研究列表）
- **链接：** https://x2robot.com/en/blog 、https://x2robot.com/en/research （中文 `/blog`、`/research` 列表一致）
- **机构：** 自变量机器人（X Square Robot），官网 About 写明 Founded in December 2023
- **核查日期：** 2026-10-09
- **方法：** 读取列表页 HTML 与 Next.js 页面数据中的标题、日期与链接；列表页未见分页或"加载更多"。
- **用途：** 确认公司路线「自变量机器人」覆盖全部官方技术博文。

## 官网列表（2026-10-09 共 8 篇）

| 栏目 | 官网日期 | 标题 | 官网入口 | 本库节点 |
| --- | --- | --- | --- | --- |
| Blog | 2026-09-02 | TwinDEX | `/pages/twindex` | [TwinDEX](../../wiki/entities/twindex.md) |
| Blog | 2026-08-27 | WALL-SS | `/pages/ss` | [WALL-SS](../../wiki/entities/paper-wall-ss.md) |
| Blog | 2026-08-03 | Human-to-robot One-Shot Skill Acquisition (HOST) | `/pages/host` | [HOST](../../wiki/entities/paper-host-one-shot-human-video.md) |
| Blog | 2026-06-30 | X-Tokenizer | `/pages/x-tokenizer` | [X-Tokenizer](../../wiki/entities/cn-os-x-tokenizer.md)（本次由占位页补为完整详情） |
| Blog | 2026-06-10 | XRZero-G0 | `/x2go` | [XRZero-G0](../../wiki/entities/xrzero-g0.md)（本次新建） |
| Blog | 2026-05-29 | WALL-WM | `/pages/wm` | [WALL-WM](../../wiki/entities/paper-rcl-2606-01955-wall-wm-carving-world-action-modeling-at-the-eve.md)（本次由清单索引补为完整详情） |
| Research | 2026-05-28 | WALL-OSS-0.5 | `/oss` | [WALL-OSS-0.5](../../wiki/entities/paper-wall-oss-0-5.md)（本次新建） |
| Research | 2025-09-08 | WALL-OSS | `/research/68bc2cde8497d7f238dde690` | [WALL-OSS / WALL-X](../../wiki/entities/cn-os-wall-x.md)（本次由占位页补为完整详情） |

## 列表外的相关入口

- **Medium（@Xsquarerobot）：** RSS 仅 1 篇 2025-12-05 行业评论（VLA 记忆），不是模型或系统发布，未纳入路线。
- **arXiv：** [X2Streaming-TTS](../../wiki/entities/paper-x2streaming-tts.md)（arXiv:2608.18661）为语音合成研究，不在官网列表，未纳入路线。
- **News 栏目：** 公司新闻与活动，不计入技术博文。

## 日期口径

公司路线优先用论文 v1 或官方发布日；官网列表日期晚于 arXiv v1 时（XRZero-G0、X-Tokenizer、HOST），路线节点按 arXiv 月份并在 `date_note` 中注明官网日期。
