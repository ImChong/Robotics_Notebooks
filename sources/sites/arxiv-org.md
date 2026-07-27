# arXiv.org — 开放获取学术预印本档案

> 来源归档

- **标题：** arXiv.org e-Print archive
- **类型：** site（开放获取预印本平台 / 独立非营利组织）
- **来源：** arXiv（2026 年起独立非营利；此前长期由康奈尔大学托管）
- **链接：** <https://arxiv.org/>
- **关于页：** <https://info.arxiv.org/about/>
- **API：** <https://info.arxiv.org/help/api/index.html>
- **入库日期：** 2026-07-27
- **一句话说明：** 面向物理、数学、计算机科学（含 **cs.RO Robotics**）、统计、电气工程等学科的 **免费开放获取预印本档案与分发服务**；材料 **未经 arXiv 同行评审**，仅经主题与学术价值 moderation。

## 为什么值得保留

- **本库主文献入口**：`sources/papers/*_arxiv_*.md` 与大量 `wiki/entities/paper-*.md` 以 arXiv `abs` / `pdf` / `html` 为默认可引用预印本；缺少平台级节点会导致「发表渠道」只覆盖顶会顶刊、漏掉日常检索层。
- **与 peer-reviewed 渠道正交**：ICRA / CoRL / T-RO 等决定 **录用与最终版本**；arXiv 决定 **最早公开、可机器检索、可 API 拉取** 的预印本层。
- **工程可互操作**：公开 API、标识符方案、批量数据与 OAI 接口，支撑文献工具、站内 ingest 与外部索引（DBLP、Semantic Scholar 等）对接。

## 官方定位摘录（About，2026-07 核查）

- **开放获取 + 策展**：任何人可使用的研究共享平台；由志愿学科 moderator 社区策展。
- **服务范围**：投稿、编译/生产、检索与发现、面向人的 Web 分发、面向机器的 API，以及内容策展与长期保存。
- **学科覆盖**：physics、mathematics、computer science、quantitative biology、quantitative finance、statistics、electrical engineering and systems science、economics。
- **治理与经费**：董事会治理；日常由 CEO 与员工执行；志愿者 moderator + Editorial / Institutions / Science Advisory Council；资助来自 Simons Foundation International、成员机构与捐赠者。
- **审稿边界**：投稿经 **moderation**（主题归类 + 学术价值检查）；**内容不经 arXiv 同行评审**，文责在投稿者；托管不构成对方法/结论的背书。
- **历史节点**：1991 年 Paul Ginsparg 创办；长期与康奈尔大学合作托管；**2026 年**确立为 **独立非营利组织**。

## 与机器人研究相关的学科入口（首页摘录）

| 档案 / 分类 | 路径提示 | 与本库关系 |
|-------------|----------|------------|
| **cs.RO** | Computer Science → Robotics | 机器人方法/系统预印本主分类 |
| **cs.LG / cs.AI / cs.CV** | CoRR 机器学习 / AI / 视觉 | VLA、世界模型、模仿学习常见旁类 |
| **eess.SY / cs.SY** | Systems and Control | 控制、估计、系统辨识交叉 |
| **stat.ML** | Statistics → Machine Learning | 学习理论与方法预印本 |

首页声明规模量级约为 **近 240 万** 篇；About 页同步写「超过三百万」——以官网当前文案为准，本页不把数字当作稳定 KPI。

## URL 与标识符（工程常用）

| 形态 | 示例 | 用途 |
|------|------|------|
| abs | `https://arxiv.org/abs/<id>` | 元数据与摘要页（本库论文元数据默认链） |
| pdf | `https://arxiv.org/pdf/<id>` | 可下载 PDF |
| html | `https://arxiv.org/html/<id>` | 部分版本的 HTML 全文 |
| 版本 | `…/abs/<id>vN` | 修订追踪；引用时宜固定版本或注明「最新」 |

标识符常见形式：`YYMM.NNNNN`（新式）或 `archive/YYMMNNN`（旧式分类前缀）。

## API 与品牌约束（官网摘要）

- 公开 API 面向互操作；商业项目须先读 API Terms、品牌使用指南，并考虑 affiliate。
- 独立非商业项目可使用公共 API，但 **不得** 以 arXiv / arXiv.org / arXiv Labs 等名称或标识暗示官方背书；建议致谢语：*Thank you to arXiv for use of its open access interoperability.*
- 需要官方深度协作时可申请 **arXivLabs**。

## 开源 / 数据开放状态（步骤 2.5）

- **平台本身**：Web + 公开 API + 批量数据管线 → **开放获取基础设施**（非「论文代码仓库」语义）。
- **单篇论文代码**：以各论文项目页 / GitHub 为准；**不能**从 arXiv 托管推断实现已开源。

## 与本仓库现有资料的关系

| 资料 | 关系 |
|------|------|
| [robotics-venues-primary-refs.md](robotics-venues-primary-refs.md) | 顶会顶刊 **一手投稿/论文集** 入口；arXiv 是其上游/并行的预印本层 |
| `sources/papers/*_arxiv_*.md` | 几乎全部论文摘录以 arXiv ID 命名与外链 |
| [wiki/comparisons/robotics-research-venues.md](../../wiki/comparisons/robotics-research-venues.md) | 发表渠道选型；应与本平台区分「预印本 vs 录用版本」 |

## 对 wiki 的映射

- 升格实体：[wiki/entities/arxiv.md](../../wiki/entities/arxiv.md)
- 交叉对比：[wiki/comparisons/robotics-research-venues.md](../../wiki/comparisons/robotics-research-venues.md)
- 总览入口：[wiki/overview/robot-learning-overview.md](../../wiki/overview/robot-learning-overview.md)
