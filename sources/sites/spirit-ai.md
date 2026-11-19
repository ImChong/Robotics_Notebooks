# 千寻智能（Spirit AI / Spirit.AI）官网、文档、新闻时间线与开源核查

> 来源归档（site / 官方站 + 开发者文档 + GitHub / HuggingFace + 权威媒体系审计）

- **主站 / 博客入口：** <https://www.spirit-ai.com/blog/>（首页 <https://www.spirit-ai.com/>；英文站 <https://www.spirit-ai.com/en/>）
- **公司介绍与产品：** <https://www.spirit-ai.com/about> · <https://www.spirit-ai.com/product> · <https://www.spirit-ai.com/news>
- **开发者文档：** <https://docs.spirit-ai.com/zh/>（最新 v2.7，历史 v2.6；Moz1 说明书、MozRobot SDK、仿真与遥操作指南）
- **GitHub 组织：** <https://github.com/Spirit-AI-Team>（5 个公开仓）
- **HuggingFace 组织：** <https://huggingface.co/Spirit-AI-robotics>（2 个模型仓，0 个公开数据集仓）
- **工商主体与办公地点（官网口径）：**
  - 官网 `/news/8` 记「2024年1月，千寻智能（杭州）科技有限公司正式注册成立」，文末又附「万境千寻（北京）科技有限公司，统一社会信用代码 `91110108MADAJAR02C`」；`/news/40` 提及「千寻智能（北京）科技股份有限公司」。
  - `/about` 列三地办公：杭州（余杭区阿里巴巴数字生态创新园）、北京（海淀区中关村软件园互联网创新中心）、深圳（南山区「模力营」AI 生态社区）；ICP 备案 `京ICP备2025103204号`。
  - `/about` 与 `/news/8` 记成立日为 **2024 年 1 月**；同期早先新闻稿 `/news/6`、`/news/7` 曾写 **2024 年 2 月**。
- **核查日期：** 2026-11-19
- **抓取与核对方式：** 直接读取 `www.spirit-ai.com` 的 `/about`、`/product`、`/blog/`（含 Vite chunk 数据）、`/news` 及关键新闻页（`/news/5,6,7,8,10,11,22,24,26,28,33,34,36,37,38,40,41,42,49` 与英文 `/en/news/1,12,13,23,25,27,30,48`）；读取 `docs.spirit-ai.com/zh/` 索引与 `moz1/product-intro`、`open-resources/pi`；通过 GitHub API 与 HuggingFace API 查询 `Spirit-AI-Team` 与 `Spirit-AI-robotics` 仓库清单与许可证；检索 SCMP、TNW、36氪等对 2026 年 6–8 月 RoboArena 榜单争议的公开报道与千寻回应。
- **核对范围：** 通读官网 `/about`、`/product`、`/blog/`（仅 `spirit-v1-5` 一篇技术博文）与 `/news` 全部 27 条中文新闻条目及核心详情全文，重点精读 `/news/41`（预训练负责人郭俊良全栈技术分享）、`/news/8`（公司与产品时间线）与 `/news/49`（WAIC/运动会）；核对开发者文档 `v2.7` 目录与 Pi0.5 适配页；核查 GitHub / HF 公开资产边界与 RoboArena 审计记录。
- **通读：** 已完成
- **覆盖 wiki：** [千寻智能（Spirit.AI）](../../wiki/entities/spirit-ai.md) · [Spirit-v1.5](../../wiki/entities/cn-os-spirit-v1-5.md)

---

## 核心团队与公司定位（官网 `/about`、`/news/8`）

- **创始人兼 CEO：韩峰涛**（十年以上机器人行业研发与商业化经验，官网称曾带领团队研发交付工业与协作机器人产品近两万台）。
- **联合创始人兼首席科学家：高阳**（UC Berkeley 博士，师从 Pieter Abbeel；清华大学交叉信息研究院助理教授；代表研究包括 EfficientZero / EfficientZero v2、ViLa、CoPa，以及千寻团队后续具身系列论文）。
- **联合创始人兼 COO：郑灵茵**（工业机器人出海与市场商业化背景）。
- **预训练负责人：郭俊良**（代表公司主讲「Building Generalist Robot Policies —— from Data to Deployment」全栈架构，亦为 [Legato](../../sources/papers/legato_arxiv_2602_12978.md) 项目负责人）。
- **核心使命与愿景（`/about`）：** 愿景「缔造新一代智慧劳动力」；使命「10年，让全球10%的人拥有自己的机器人」；聚焦「通用具身大脑 + 真实世界数据闭环 + 全力控本体」。

---

## 博客目录核查（`https://www.spirit-ai.com/blog/`）

- 截至 2026-11-19 检查 `/blog/` 打包索引脚本与详情表，`https://www.spirit-ai.com/blog/` 当前仅发布 **1 篇** 深度技术博客：
  - [Spirit-v1.5: Clean Data Is the Enemy of Great Robot Foundation Models](https://www.spirit-ai.com/blog/spirit-v1-5)（发布日 **2026-01-11**）→ 完整归档见 [spirit_ai_spirit_v1_5_blog_2026-01-11.md](../blogs/spirit_ai_spirit_v1_5_blog_2026-01-11.md)。

---

## 官网新闻与里程碑时间线

| 日期 | 事件 | 依据链接与说明 |
|------|------|----------------|
| 2024-01 | 千寻智能（杭州）科技有限公司注册成立（部分早期通稿记 2024-02） | `/about` · `/news/8`（注：`/news/6`、`/news/7` 曾写 2024-02） |
| 2024-07 | 初代机器人 Moz 0 诞生 | `/news/8` |
| 2024-08 / 11 | 近 2 亿元种子轮+天使轮（弘晖等）与数千万元天使+轮（柏睿资本） | `/news/8` |
| 2025-03 | 发布 Spirit v1 VLA 模型抢先版；完成 5.28 亿元 Pre-A 轮融资（Prosperity7 领投） | `/news/8` · `/en/news/1`（3 月 31 日稿） |
| 2025-06 | 正式发布全力控轮式人形机器人 **Moz1**（26 DoF 全身一体化力控关节、七轴力位混合臂、全向轮式底盘） | `/news/8` · `/news/10` · `/product` · `docs.spirit-ai.com/zh/` |
| 2025-07 | 完成近 6 亿元 PreA+ 轮融资（京东领投）；携 Moz1 亮相 WAIC、WRC | `/news/8` · `/news/11` |
| 2025-12 | 宁德时代中州基地动力电池 PACK 产线规模化部署人形机器人「小墨（Moz）」替代 EOL/DCR 高压测试插接工序（自报插接成功率 >99%） | `/news/22`（`/upfiles/news/202512/`）· `/en/news/23` |
| 2026-01 | **Spirit-v1.5** 发布并开源代码与权重；官宣在 RoboChallenge Table30 榜单登顶（截至 2026-01-11） | `/blog/spirit-v1-5` · `/news/24` |
| 2026-02 | **HuMI**（`arXiv:2602.06643`）与 **Legato**（`arXiv:2602.12978`，后入选 RSS 2026）挂出；2 月 25 日官宣连续两轮近 20 亿元（约 2.8 亿美元）融资 | `/en/news/30` · `/news/26` · `/news/33` |
| 2026-03–05 | 宣布与博世中国（`/news/28`）、地瓜机器人（`/news/32`，Spirit v1.5 适配旭日 S600）、舍弗勒（`/news/35`）合作 | `/news` |
| 2026-06 | **Spirit v1.6**（未开源；定位 VLA + 世界模型融合）发布及 15 亿元 A+ 轮融资；6 月初官宣登顶 RoboArena 榜单，随后 7–8 月 RoboArena 方法改版后移除（见下节）；同期 **OpenHLM**（`arXiv:2606.22174`，CoRL 2026 Spotlight）发布 | `/news/34` · `/news/36` · arXiv |
| 2026-07 | 2026 WAIC（7 月 17 日开幕）首次公开发布新一代商用服务机器人 **Moz2**；同月挂出 **Emergent Transfer**（`arXiv:2607.25593`） | `/news/38` · `/en/news/48` · `/news/42` |
| 2026-08 | 8 月 25 日在第二届世界人形机器人运动会凭借搭载 Spirit v1.6 的 Moz1 获全自主推理办公场景冠军、餐饮场景亚军；Pretraining Lead 郭俊良在探月计划作 40 分钟全栈系统分享 | `/news/49` · `/news/41` |
| 2026-11 | 联合牛津、国科大、清华发布 **VGE**（`arXiv:2611.22625`，冻结 Qwen3.5-9B 引导多目标指代接地前端，代码待发表后公开） | arXiv:2611.22625 |

---

## 全栈技术架构（`/news/41` 郭俊良分享与 `/product`、`docs.spirit-ai.com`）

### 1. 横向基建与三段式 Physical AI 系统
- **横向底座：** `Platform Infrastructure`（数据采集上传 → T+1 自动化清洗与标注 → 仿真 replay 校验，尤其验证 uDAS 人类穿戴数据在仿真中的可回放性 → 模型训练 → Checkpoint 管理与下发评测 → 在线 rollout 回流）与 `AI Infrastructure`。
- **DATA（数据金字塔）：**
  1. 底层：**互联网视频**（约 30 万小时，经清洗/过滤算子保障有效性与物理常识）；
  2. 中层：**Ego-Centric 与自研可穿戴数采设备 uDAS**（迭代两年七代，真实生活非脚本多样采集累计 **超过 20 万小时**；自报在数据有效性达到遥操作 **95%** 的同时，单条成本降为遥操作的 **1/20**）；
  3. 顶层：**真机遥操作数据与 HRPI 在线部署回流（rollout 数据与 RL 微调闭环）**。
- **MODEL（Spirit Omni Model）：** 原生多模态统一架构，结合 VLA 与 World Model；输入指令、视觉与本体状态（在特定消融中探索去状态化），输出文本（Text）、运动离散词元（Motion Token）与连续动作块（Action Chunks）。
- **AGENT 层：** 负责前端情感陪伴/多模态交互（如展会现场的 Moz1/Moz2）与「模糊指令翻译」（将用户自然模糊需求转译为 VLA 可解析的结构化任务指令）。

### 2. 公开的四大研究瓶颈与对应论文（`/news/41`）
1. **Cross-Embodiment Alignment（RSS Workshop）：** 双臂 Moz-1 与单臂 Franka 混合训练时，通过 Frame / State / Arm-configuration / Visual 四维对齐，Moz-1→Franka 迁移成功率由 36.7% 升至 76.7%，Franka→Moz-1 由 30.0% 升至 85.0%，物体迁移由 68.0% 升至 96.7%（「解锁迁移的是数据语义，而不只是模型的 head」）。
2. **Point-VLA（IROS 2026，arXiv 编号在分享稿未给出）：** 当日常语言模糊（如「那瓶水」）时，显式视觉指向标记优于冗长方位文本。对应后续感知接地前端探索可见 [VGE](../../wiki/entities/paper-vge-2611-22625.md)。
3. **State-free Policy（在投，2025/2026 预印本）：** 单任务微调直接喂入本体状态易导致过拟合（抓空后仍机械走完固定位置轨迹）；在补齐腕下等充分视觉观测前提下剥离状态输入，可显著改善高度与空间泛化。
4. **Legato（RSS 2026，arXiv:2602.12978）：** 训练期流动力学重塑实现动作块原生延续，显著抑制推理延迟与跨块边界带来的模态切换和停顿（详归档见 [legato_arxiv_2602_12978.md](../papers/legato_arxiv_2602_12978.md)）。

### 3. 硬件平台与开发者文档（`/product`、`docs.spirit-ai.com/zh/`）
- **Moz1：** 轮式人形机器人（全向轮式底盘 + 高性能力位混合七轴仿生臂 + 全身 26 个一体化力控关节 + STO 安全与高精度碰撞检测；配套 `MozRobot SDK`、Isaac Sim 与 MuJoCo 仿真资产、SLAM 导航与遥操作教程；开源 `openpi` Pi0.5 微调文档明确说明：**「请下载 TOS Browser，并联系千寻售后获取秘钥，下载源码」**，即硬件适配源码面向客户定向发放而非完全公开）。
- **Moz2：** 2026-07-17 WAIC 首次公开亮相的商用服务机器人新品；官网未公开详细自由度/负载/售价规格。

---

## GitHub 与 HuggingFace 资产核查（2026-11-19）

| 平台 / 名称 | 链接 | 许可证 / 状态 | 说明与边界 |
|-------------|------|----------------|------------|
| `Spirit-AI-Team/spirit-v1.5` | <https://github.com/Spirit-AI-Team/spirit-v1.5> | **MIT License** | 含 `Qwen3-VL-4B` 骨干 + `BaseDiT` 流匹配动作头推理/RoboChallenge 评测包装及 `train.py` FSDP 微调脚本（仅含 `RoboChallengeDataset` `move_objects_into_box` 读取示例）；**不含预训练管线与预训练全量数据** |
| `Spirit-AI-robotics/Spirit-v1.5` & 微调版 | <https://huggingface.co/Spirit-AI-robotics/Spirit-v1.5> | **Apache-2.0** | 公开权重（含 base 与 `Spirit-v1.5-for-RoboChallenge-move-objects-into-box`）；**HF Datasets 数量为 0**，Diverse Collection 与 uDAS 数据均未公开 |
| `Spirit-AI-Team/issac_moz1` | <https://github.com/Spirit-AI-Team/issac_moz1> | 无显式仓库 LICENSE（API 返回 null） | Moz1 轮式人形机器人在 Isaac 仿真中的模型/仿真仓库 |
| `Spirit-AI-Team/PI_Official` / `lerobot` / `ByteCheckpoint` | `https://github.com/Spirit-AI-Team` | Apache-2.0（上游 fork / 集成） | 生态工具集成与分布式 checkpoint 工具 |
| `lyfeng001/Legato-kinetix` | <https://github.com/lyfeng001/Legato-kinetix> | **MIT License**（基于 `real-time-chunking-kinetix`） | Legato 论文仅开源 Kinetix 2D 仿真训练与评测脚本；真机 `pi0.5`/`pi0`/`GR00T N1.6` 版本未开源 |
| Spirit v1.6 / Spirit Omni Model / uDAS 管线 | 仅 `/news/34`、`/news/41`、`/news/49` 报道 | **未开源** | 暂无独立公开论文、代码仓库或权重 |

---

## 争议、限制与中立口径记录

1. **RoboArena 榜单登顶与移除争议（2026 年 6–8 月）：**
   - 2026-06-03 千寻智能公布 Spirit v1.6 在 RoboArena 取得 1,924 Elo、一度位居榜首并同期宣布 15 亿元 A+ 轮融资（`/news/34`）。
   - 据 SCMP（2026-08-04）与多家媒体报道，RoboArena 组织方随后审计发现去中心化评测规模尚小（不足 30 家机构、约 3,900 次评测）且部分匿名志愿者评测轨迹引发异常争议，升级榜单方法并重整官方榜，将 Spirit v1.6 以及 NVIDIA Cosmos 3、X Square WALL-OSS 一并移出官方主榜。千寻智能向 SCMP 回应称：团队正与 RoboArena 沟通，后者认可模型性能表现，未列官方榜主因「未满足新版的最低跨站点样本量要求」，而非对模型能力本身的否定。
   - **百科记录准则：** 不把早期 1,924 Elo 短暂登顶或单篇公关战报当作长期客观排名定论，同时客观呈现评测协议未成熟（样本量、去中心化审计门槛）与公司回应口径。
2. **数据与产线自报口径：** uDAS「1/20 成本与 95% 有效性」「20 万小时以上穿戴采集」「宁德时代产线 >99% 成功率」「人均采集效率 +200%」均来自公司官网博文、新闻稿与高管直播分享，尚无第三方独立复现审计报告；uDAS 硬件驱动、T+1 清洗平台和全部 Diverse Collection 原始语料未对外开放。
3. **跨机构论文与商业产品边界：** HuMI（`2602.06643`）、Legato（`2602.12978`）、OpenHLM（`2606.22174`）、Emergent Transfer（`2607.25593`）、VGE（`2611.22625`）均为高阳团队 / 千寻与清华、上交、期智研究院、同济、牛津等单位联合发表的学术论文，研究验证不等于闭源商用模型 Spirit v1.6 的完整内部实现。
