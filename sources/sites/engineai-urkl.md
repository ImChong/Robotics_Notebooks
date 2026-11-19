# EngineAI URKL 赛事门户（robot-fighting-competition）

> 来源归档

- **标题：** URKL — Ultimate Robot Knock-out Legend（EngineAI 首届人形格斗联赛）
- **类型：** site（商业赛事 / 报名门户）
- **URL：** <https://en.engineai.com.cn/robot-fighting-competition.html>
- **机构：** 众擎机器人（ENGINEAI / Zhongqing Robotics；深圳）
- **入库日期：** 2026-07-20
- **核对日期：** 2026-11-19
- **核对范围：** 全文通读官方英文赛事页（`urkl_site.txt`，`robot-fighting-competition.html`）与官方 T800 产品页（`t800_site.txt`，`product-t800.html` 四版本规格对比及脚注 `[8]`）
- **通读：** 已完成
- **一句话说明：** EngineAI 以 **统一 T800 全尺寸人形平台** 举办全球首届 **标准化算法格斗联赛 URKL**：双轨「展演 + 竞技」、线上资格赛与线下淘汰赛，强调 **自主平衡与运控算法** 而非硬件魔改；官网同步开放 T800 商用四版本硬件平台。

## 官网公开要点（全文核对）

| 区块 | 内容要点 |
|------|----------|
| **品牌** | **URKL**（Ultimate Robot Knock-out Legend） |
| **赛事定位** | 世界级机器人技术创新、交流与专业竞技平台；推动格斗赛事专业化、规模化与产业化 |
| **核心使命** | Technology Empowerment · Innovation Driven · Global Collaboration |
| **赛制形态** | **双轨**：**Exhibition Showcases（展演）** + **Combat Competition（竞技）** |
| **页面状态与入口** | 官网英文门户标注 **2026.3.1** 时间锚点、`Online Qualifiers COMING SOON`、提供 **View the Rules** 与 **Register Now** 入口（截至 `urkl_site.txt` 抓包页尾倒计时显示 `-126 天`，页面版权为 `© 2025 Engineai`） |

## 统一硬件平台（T800 产品页 `t800_site.txt` 全文核对）

赛事与 [T800 产品页](https://en.engineai.com.cn/product-t800.html) 绑定。全系共通参数：**173 cm** 身高、**75–85 kg** 整备体重、**450 N·m** 峰值关节力矩、PMSM 电机 + 交叉滚子轴承（crossed roller bearing）、腿部主动散热、固态电池 **3–5 h** 续航、最大步速 **≥ 2 m/s**、8 核 CPU；产品页脚注 `[8]` 明确声明仅限民用（civilian use only）。

| 版本（Edition） | 自由度（DoF） | 算力模块 | 传感 / 通信 / 二次开发 |
|---|---|---|---|
| **BASIC EDITION** | **25 DoF** | Orin NX 16G | 无 3D LiDAR；不支持二次开发 SDK |
| **OPEN SOURCE EDITION** | **25 DoF** | AGX Orin 64G（支持升级 Thor） | 配备 3D LiDAR；支持二次开发（Secondary Development √） |
| **PRO EDITION** | **29 + 14 = 43 DoF**（含双 7-DoF 触觉灵巧手，负载 5 kg） | AGX Orin 64G（支持升级 Thor） | 配备 3D LiDAR + 触觉灵巧手；支持二次开发 |
| **MAX EDITION** | **32 + 14 = 46 DoF**（含双 7-DoF 触觉灵巧手） | AGX Orin 64G（支持升级 Thor） | 在 PRO 基础上新增 **5G** 模块；支持二次开发 |

注：此前入库记录的参考售价区间为 Basic **$40,500**、Open Source Edition **$54,000**（`t800_site.txt` 英文规格表页面未直接标注价格，保留作历史参考线索）。

## 第三方公开信息（交叉核对，非官网原文）

| 主题 | 要点 | 来源 |
|------|------|------|
| **赛制模型** | **标准化硬件 + 差异化算法**；禁止暴力改装；鼓励运控、感知决策与非破坏性结构防护 | [Humanoids Daily](https://www.humanoidsdaily.com/news/engineai-opens-global-registration-for-urkl-the-1-4-million-race-for-humanoid-supremacy) |
| **参赛对象** | 全球高校、企业与研究机构；筛选后 **16 队** 进正赛（另有报道首场比赛 **32 队**） | 同上；[Global Times](https://www.globaltimes.cn/page/202607/1366175.shtml) |
| **时间线** | 线上报名 **2026-03-01 ~ 04-30**；四月线上资格赛；**05–07** 现场预赛；**10–11** 十六强/八强；**12–2027-01** 总决赛 | Humanoids Daily |
| **单局规则** | **BO3**；每局净比赛 **5 min**；倒地 **10 s** 内自主起身不罚；否则 **manual reset**（每场最多 **2 次**） | Humanoids Daily |
| **奖项** | 冠军 **10 kg 纯金腰带**（约 **1000 万元 RMB / $1.45M**）；前 16 名获赠 **T800 整机**；前 8 队员可进 EngineAI **招聘 fast-track** | Humanoids Daily |
| **首秀** | **2026-07-16** 深圳南山文体中心开幕战；**T800** 真机互殴，含 **头部击飞后躯干仍可继续出拳** 等耐久演示 | Global Times；[Office Chai](https://officechai.com/ai/china-hosts-humanoid-robot-combat-tournament-thats-mma-with-robot-fighters/) |
| **产业对照** | 与西方 **REK**（VR 真人 pilot + Unitree G1）形成「**自主算法联赛 vs 人类遥操作联赛**」对照轴 | 本仓库 [rek-com.md](./rek-com.md) |

## 源码 / 数据开放核查（步骤 2.5）

| 类别 | 结论 |
|------|------|
| **赛事页** | 截至 **2026-07-20** 未列 GitHub / Hugging Face / 数据集下载；第三方导读 [urkl.org](./urkl-org.md) 仍记英文页为 **Coming Soon**（复核至 2026-07-23） |
| **T800 产品** | 提供 **Open Source Edition** SKU（官网标「secondary development: Support」），属 **商业硬件 + SDK** 而非公开训练代码仓库 |
| **官方 FAQ（2026-07-24）** | [wechat_urkl_faq_01.md](../blogs/wechat_urkl_faq_01.md) 宣称将 **开源本届赛事相关代码**，但 **未给仓库 URL** → **宣称将开源 / 待发布** |
| **归类** | **商业赛事 + 硬件平台**；参赛队算法是否强制开源 **未在官网细则中写死**；组委会侧「赛事相关代码」开源 **待链接落地** |

## 对 wiki 的映射

- 主实体：[URKL（EngineAI 人形格斗联赛）](../../wiki/entities/urkl.md)
- 硬件交叉：T800（本 ingest 未单独建实体页；见产品页与 [LHBS](../../wiki/entities/paper-notebook-learning-human-like-badminton-skills-for-humanoi.md) 中的 EngineAI PM01 对照）
- 任务/路线交叉：[Teleoperation](../../wiki/tasks/teleoperation.md)、[人形拳击纵深路线](../../roadmap/depth-humanoid-boxing.md)
- 产业对照：[REK](../../wiki/entities/rek.md) — VR 遥操作格斗联赛
- 姊妹归档：[urkl-org.md](./urkl-org.md)（独立导读）、[wechat_urkl_faq_01.md](../blogs/wechat_urkl_faq_01.md)（官方 FAQ）

## 待后续深读（可选）

- [ ] 官方规则 PDF / 评分细则全文（击打精度、闪避、起身速度等权重）
- [ ] 线上资格赛评测接口与提交格式（开放后）
- [ ] T800 SDK/API 文档是否对参赛队单独开放
- [ ] 组委会「本届赛事相关代码」开源仓库 URL 落地后回写步骤 2.5