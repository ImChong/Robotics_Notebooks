# DeepTutor 官方站（deeptutor.info）

> 来源归档（ingest）

- **类型：** 网站 / 产品页 / 文档站
- **入口：** <https://deeptutor.info/>
- **代码仓：** <https://github.com/HKUDS/DeepTutor>
- **技术报告：** <https://arxiv.org/abs/2604.26962>
- **技能生态：** <https://eduhub.deeptutor.info/>
- **收录日期：** 2026-08-31
- **抓取说明：** 以 **2026-08-31** 对 deeptutor.info 首页与仓库 README 交叉核对为准；版本号与能力列表随 release 演进，勿在 wiki 固化具体 release 细节。

## 一句话

**deeptutor.info** 是 DeepTutor 的 **官方文档与产品导览面**：四种安装路径（PyPI / Docker / Source / CLI）、八大工作区能力说明，以及 EduHub 技能生态入口。

## 开源状态（步骤 2.5）

- **已开源：** 站点链到 GitHub 仓、arXiv 论文与 PyPI `deeptutor`；Docker 镜像 `ghcr.io/hkuds/deeptutor` 公开发布。
- **边界：** EduHub 为独立 registry 服务；LLM / embedding / search 提供商密钥由用户自备。

## 为什么值得保留

- 与 [仓库源归档](../repos/hkuds_deeptutor.md) 配对，区分 **营销/文档面（站点）** 与 **实现仓（GitHub）**。
- 首页按 **I–VIII 八大表面**（Home、Partners、My Agents、Co-Writer、Book、Learning Space、Memory、Knowledge Center）组织，便于 wiki 实体页「流程总览」对齐。
- 安装向导强调 **workspace 目录 + `deeptutor init` + `deeptutor start`**，与 CLI-only 路径分开展示。

## 公开要点（归纳）

| 区块 | 内容 |
|------|------|
| **Hero** | Agent-native Learning Companion；Fully Open-Sourced |
| **八大表面** | Chat 默认环、Partner IM、子代理 consult、Co-Writer、交互式 Book、Learning Space、三层 Memory、多引擎 Knowledge Center |
| **安装 I–IV** | PyPI 推荐；Docker GHCR；源码 dev；CLI headless（`packaging/deeptutor-cli`） |
| **EduHub** | 教学向 skill registry；`deeptutor skill search/install/publish` |
| **Collaborate** | 站内设链至合作入口 |

## 对 wiki 的映射

- 升格页面：[wiki/entities/deeptutor.md](../../wiki/entities/deeptutor.md)
- 仓库侧归档：[sources/repos/hkuds_deeptutor.md](../repos/hkuds_deeptutor.md)
- 论文归档：[sources/papers/deeptutor_arxiv_2604_26962.md](../papers/deeptutor_arxiv_2604_26962.md)
