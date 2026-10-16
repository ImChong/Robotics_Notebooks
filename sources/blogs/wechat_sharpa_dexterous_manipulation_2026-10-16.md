# 硬件、数据到模型：Sharpa如何系统性推进灵巧操作

> 来源归档（blog / 微信公众号）

- **标题：** 硬件、数据到模型：Sharpa如何系统性推进灵巧操作
- **类型：** blog
- **作者：** 机器之心（编辑 Panda、Youli；采访 Sharpa 学术负责人张凯峰）
- **原始链接：** https://mp.weixin.qq.com/s/wDRJgJArppWbAWm1QmzwNQ
- **发表日期：** 2026-10-16（微信页 `createTime`）
- **入库日期：** 2026-10-16
- **抓取方式：** 直接请求页面 HTML 后本地抽取正文（`js_content`），未经第三方阅读器。
- **原始落盘：** [`sources/raw/wechat_sharpa_dexterous_2026-10-16/article.md`](../raw/wechat_sharpa_dexterous_2026-10-16/article.md)
- **一句话说明：** 以 Sharpa 学术线为主线，梳理灵巧操作的数据两难（遥操作缺触觉、人类视频缺力）、三条研究主线（具身鸿沟 / 运动与交互解耦 / 以触觉为锚的跨本体表征）与 System 0/1/2 分层架构，并列出 Sharpa 参与的一组论文。
- **项目页 / 源码：** 文内仅给论文地址；公司级产品信息见 [Sharpa 实体页](../../wiki/entities/sharpa-robotics.md)。

## 文中论文与本库节点映射

每篇论文各自一个详情节点；与库内已有节点同一篇时复用，不另建。

| 论文（文内名） | arXiv | v1 日期 | 本库节点 |
|----------------|-------|---------|----------|
| SaTA（Spatially-anchored Tactile Awareness） | 2510.14647 | 2025-10-16 | [paper-sata](../../wiki/entities/paper-sata.md) |
| Tacmap | 2602.21625 | 2026-02-25 | [paper-tacmap](../../wiki/entities/paper-tacmap.md) |
| EgoScale | 2602.16710 | 2026-02-18 | [已有实体页](../../wiki/entities/paper-sa-2602-16710-egoscale-scaling-dexterous-manipulation-with-div.md) |
| DexEMG | 2603.05861 | 2026-03-06 | [paper-dexemg](../../wiki/entities/paper-dexemg.md) |
| MoDE-VLA | 2603.08122 | 2026-03-09 | [已有实体页](../../wiki/entities/paper-sa-2603-08122-mode-vla-mixture-of-dexterous-experts-vla.md) |
| FTP-1 | 2606.13102 | 2026-06-11 | [paper-ftp-1](../../wiki/entities/paper-ftp-1.md) |
| T-Rex | 2606.17055 | 2026-06-15 | [已有实体页](../../wiki/entities/paper-trex-tactile-reactive-dexterous-manipulation.md)（文内无 arXiv 链接，按标题与数据规模对应） |
| Do as I Do | 2606.19333 | 2026-06-17 | [paper-do-as-i-do](../../wiki/entities/paper-do-as-i-do.md) |
| Play2Perfect | 2606.26428 | 2026-06-24 | [paper-play2perfect](../../wiki/entities/paper-play2perfect.md) |
| Dex-X | 2609.07747 | 2026-09-07 | [paper-dex-x](../../wiki/entities/paper-dex-x.md) |
| GALATEA | 2609.10050 | 2026-09-09 | [已有实体页](../../wiki/entities/paper-galatea.md) |
| TacBPM | 2609.18174 | 2026-09-16 | [paper-tacbpm](../../wiki/entities/paper-tacbpm.md) |
| UVTA | 2609.34182 | 2026-09-28 | [已有节点](../../wiki/entities/paper-uvta-human-tactile-dexterous-manipulation.md)（来自分支 ingest/uvta-2609-34182 合并） |
| SimToolReal | 2602.16863 | 2026-02-18 | [已有实体页](../../wiki/entities/simtoolreal.md)（文内无 arXiv 链接，按标题与 Sharpa 手对应） |

**未建节点的提及（非本文论点的论文或产品）：**
- CraftNet（CES 2026 演示的三层分层模型）：文内无论文标识；库内 [WM-Craftnet](../../wiki/entities/paper-wm-craftnet.md)（arXiv 2609.07002）是另一套方法，不等同。
- IMCopilot：作为 MoDE-VLA 的底层手内技能出现，未单列论文。
- FELT、AnyTeleop、DexCap、DexUMI 等：作为对照或背景提及，不属于 Sharpa 参与的论文主线。
- Gemini 4 Argon 演示：产品新闻，非论文。

## 核心摘录（归纳，非全文）

- **数据两难**：遥操作数据与机器人本体对应但操作员感受不到接触；以人为中心的采集（DexCap、DexUMI 等）数据更自然，但人手状态不等于机器人动作，缺力与触觉。
- **三条主线**：(1) 击穿具身鸿沟，让人类数据成为动作燃料（UVTA、DexEMG、Dex-X）；(2) 解耦运动与交互，接触交给仿真与原子技能（MoDE-VLA 的脚踏板 + IMCopilot、Dex-X 点云接口、TacBPM 隐空间接口，Do as I Do、SimToolReal、Play2Perfect 为他队对照）；(3) 以触觉为锚的跨本体表征（FTP-1、EgoScale、T-Rex）。
- **分层架构**：System 2（VLM，约 1 Hz）、System 1（运动，约 10 Hz）、System 0（交互，约 100 Hz，实时触觉）。
- **硬件平台**：Sharpa Wave（22 DoF，1:1 仿生手）被多个团队用作研究平台；IROS 2026 发布 D01 整机、W02 灵巧手、AE01 触觉反馈数据手套（文内）。
- **真实部署**：Sharpa 与 Dairy Queen 合作的机器人餐厅 2026-08-29 于上海吴江路开业，55 步流程（文内，据报道 98% 步骤涉及触觉）。
- **判断**：文内张凯峰估计单刚体类人灵巧操作 3–6 个月内有突破，整体约两年尺度（个人判断，非公开数据）。

## 与本库其他页面的关系

- 灵巧操作与触觉：[接触丰富操作](../../wiki/concepts/contact-rich-manipulation.md)、[视触觉融合](../../wiki/concepts/visuo-tactile-fusion.md)
- 数据路线：[EgoScale 方法页](../../wiki/methods/egoscale.md)、[仿真到现实](../../wiki/concepts/sim2real.md)

## 参考来源

- [硬件、数据到模型：Sharpa如何系统性推进灵巧操作（原始抓取）](../raw/wechat_sharpa_dexterous_2026-10-16/article.md)
- 公司背景检索（2026-10-16）：[PR Newswire 经 Utusan 转载的 Sharpa 量产新闻稿](https://www.utusan.com.my/pr-newswire-en/2025/12/ai-robotmaker-sharpa-reaches-key-milestone-with-mass-production-of-worlds-most-advanced-human-sized-robotic-hand/)
