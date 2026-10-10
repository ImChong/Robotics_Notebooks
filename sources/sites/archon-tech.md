# 源策未来（Archon Robotics）官网核查

> 来源归档（site / Archon Robotics 官方）

- **URL：** <https://www.archon.tech/>（博客列表 <https://www.archon.tech/blog>；招聘 <https://careers.archonrobotics.tech>）
- **机构：** Archon Robotics 源策未来；页脚「Archon Robotics 源策未来 © 2026」；静态资源域名 `assets.archonrobotics.tech`
- **核查日期：** 2026-10-10
- **抓取方式：** `curl -sSL -A "Mozilla/5.0"` 读取首页、`/blog`、`/blog/whole-body-intelligence`、`/blog/whole-body-intelligence-cn`。站点是 Next.js 服务端直出文字，图片和视频懒加载（页面显示 "Loading media..."）；从 HTML 中提取 `img` / `video` 地址。招聘站是客户端渲染，用 Playwright（Chromium headless）读取 `document.body.innerText`。YouTube 元数据用 oEmbed 接口读取，观看页返回 429，未取得上传日期。
- **覆盖 wiki：** [源策未来（Archon Robotics）](../../wiki/entities/archon-robotics.md)、[Archon 全身智能（WBI）](../../wiki/entities/archon-whole-body-intelligence.md)
- **社媒（页脚）：** X <https://x.com/archon_robotics>；LinkedIn <https://www.linkedin.com/company/archon_robotics>；YouTube <https://www.youtube.com/@archon_robotics>
- **联系邮箱：** Cloudflare 邮箱混淆，未解码；招聘站写 careers@archon.tech

## 首页逐段记录

### 标语

- 「Archon Robotics 源策未来」
- 「From Origin, Through Action, Toward the Future.」/「以源为始 · 以策为径 · 通向未来」
- 首屏视频：`https://assets.archonrobotics.tech/demo/20260930/landing.mov`（路径含 `20260930`，推测首页视频约在 2026-09-30 更新）

### Whole-Body Intelligence（四条要点，自报）

1. Whole-Body Intelligence framework for building generalist humanoid foundation models.
2. Context-aware behavioral foundation model for precise, stable loco-manipulation with terrain interaction.
3. Pre-training on large-scale whole-body human data builds rich interaction priors.
4. Efficient post-training with real-robot data turns pretrained knowledge into skills.

「Full video」链接：<https://youtu.be/b0h9oC8FhpU>。oEmbed 标题 *Building Whole-Body Intelligence for the Real World. We're Just Getting Started.*，频道「Archon Robotics 源策未来」；上传日期未取得。

### Mission（原文摘录）

> At Archon Robotics, we're building a Humanoid Foundation Model (HFM) for whole-body intelligence - the foundational AI layer that lets humanoid robots move, perceive, reason, and act in the real world.
>
> We aim to become a leading embodied AI company and build the "OpenAI for Humanoid Robotics" in China.

「Featured Research」只链接到 `/blog/whole-body-intelligence`。

### Team

- 团队规模：「around 20 researchers and engineers」，来自 HKU、清华、上海交大和科技公司；含前自动驾驶负责人、机器人研究者、大模型工程师、基础设施专家。

| 人物 | 官网职位 | 官网简介（自报） |
|------|----------|------------------|
| Hongyang Li 李弘扬 | Founder & Chief Robot Officer (CRO) | 端到端自动驾驶 UniAD 负责人（CVPR 2023 Best Paper）；2026 RSS Early Career Spotlight |
| Tianyu Li 李天羽 | Co-Founder & CEO | 华为 ADS 4.0 核心架构师，端到端自动驾驶量产上车；WAIC Rising Star 2026 |
| Li Chen 陈立 | Co-Founder & Head of AI | UniAD 第一作者；研究具身智能与世界模型；ECCV 2026 Area Chair |

### Investors

文字：「We brought together top-tier capital firms and leading research institutions」。Logo 依次链接：

| 顺序 | 名称 | 链接 |
|------|------|------|
| 1 | ZhenFund 真格基金 | <https://zhenfund.com/> |
| 2 | Gaorong 高榕创投 | <https://www.gaorongvc.com/> |
| 3 | IDG Capital | <https://idgcapital.com/> |
| 4 | 5Y Capital 五源资本 | <https://www.5ycap.com/> |
| 5 | Gobi 戈壁创投 | <https://www.gobivc.com/> |
| 6 | The University of Hong Kong | <https://www.hku.hk/> |
| 7 | MiraclePlus 奇绩创坛 | <https://www.miracleplus.com/> |
| 8 | SII 上海创智学院 | <https://www.sii.edu.cn/> |

官网未写轮次、金额、领投方、成立日期或总部地址。

## Blog 列表（2026-10-10 共 1 篇）

| 编号 | 标题 | 日期 | 链接 |
|------|------|------|------|
| 01 | Whole-Body Intelligence: The Pretraining Path to Large Humanoid Models | 2026-07-13 | <https://www.archon.tech/blog/whole-body-intelligence>（中文版 `/blog/whole-body-intelligence-cn`） |

详细摘录见 [archon_whole_body_intelligence.md](../blogs/archon_whole_body_intelligence.md)。

## 招聘站（careers.archonrobotics.tech）

- 文案：人形机器人的下一阶段「以规模化人类数据为基础，构建能够随真实部署持续学习和迭代的全身智能 (Whole-Body Intelligence, WBI)」。
- 「源策新星计划 / Archon Star Program」：面向全球招募研究与工程人才，不设专业背景限制。
- 未列办公地点。

## 开源渠道核查（2026-10-10）

| 渠道 | 结果 | 方法 |
|------|------|------|
| GitHub 组织 `ArchonRobotics`（显示名「Archon Robotics 源策未来」，user id 287222724） | **0 个公开仓库**，无公开成员 | GitHub 用户搜索（MCP）+ WebFetch 组织页；仓库搜索 API 返回 502，`raw.githubusercontent.com/ArchonRobotics/.github/main/profile/README.md` 404 |
| Hugging Face 组织 `ArchonRobotics`（「Archon Robotics 源策未来」） | 0 模型 / 0 数据集 / 0 Space / 0 论文；1 名成员 | `huggingface.co/api/organizations/ArchonRobotics/overview` 与 models / datasets / spaces API；组织 ObjectId 时间戳折算 2026-05-23（推测为创建时间） |
| 官网 | 无 GitHub / HF / 论文链接 | 首页与博客 HTML |
| 已发布代码的署名论文 | RoboNaldo 代码在 **OpenDriveLab** 组织（`OpenDriveLab/RoboNaldo`、`OpenDriveLab/RoboNaldo_Deploy`），不在公司组织 | 见 [robonaldo.md](../repos/robonaldo.md) |

## 相关官方页面（OpenDriveLab 站点）

- 「Archon & OpenDriveLab at RSS 2026」<https://opendrivelab.com/rss2026>（2026-10-10 读取）：RSS 2026（2026-07-13 至 17，悉尼）联合页面。列出 RISE、EgoHumanoid、GuidedVLA（主会）与 RoboNaldo、SparseVideoNav、NativeMEM（Workshop）；演讲：李天羽（CEO, Archon Robotics；07-13，*Scaling Whole-Body Humanoid Skills with Human Demonstration*）、李弘扬（HKU 助理教授 / Founder, Archon Robotics；07-14，*Whole-body Intelligence with Human-centric Data at Scale*，RSS Early Career Spotlight）、陈立（Head of AI, Archon Robotics；07-17，*Improving Embodied Policy with Compositional World Model*）；现场 RoboNaldo 人形足球演示。
