# 诺亦腾机器人：官方技术路线与公开资料

- **类型：** site / company / motion capture / embodied data
- **官网：** <https://noitomrobotics.com/>（[中文官网](https://noitomrobotics.cn/)）
- **研究与技术文章：** [NEAR Lab](https://noitomrobotics.com/research/) · [Tech Blog](https://noitomrobotics.com/tech-blog/)
- **机构：** 诺亦腾机器人（Noitom Robotics）
- **核查日期：** 2026-11-21
- **核对范围：** 中英文官网的业务与 Adam-U 案例、NEAR Lab 项目索引、2024 年诺亦腾科技 Isaac 案例全文、Adam-U 官方发布全文、ModalityNet 技术博客 Episode 1 / 2 全文、ModalityNet v1.0 的三类数据说明与格式目录、HiPHI 发布公告与数据许可。研究论文沿用各自的原文归档。
- **通读：** 已完成

## 主体与日期口径

诺亦腾机器人于 2025 年分拆成立，见[创始人访谈](https://www.qimingvc.com/en/node/7706)与[奥飞娱乐 2025 年年报](https://disc.static.szse.cn/disc/disk03/finalpage/2026-04-28/3e4d5414-270f-45a1-99a4-efaa884f5d84.PDF)。2024 年应用由北京诺亦腾科技有限公司披露，作为团队的技术前序。

| 日期 | 资料 | 核查要点 |
|---|---|---|
| 2024-09-14 | [诺亦腾科技：Isaac 与动作捕捉数据工厂](https://www.noitom.com.cn/news/20240914001.html) | PN Studio 惯性动捕、HybridTrack 光惯混合追踪和 MocapApi 对接 Isaac；文章列出真人直采、仿真遥操作、真机遥操作三种数据生产方式。 |
| 2025-07 | [Adam-U 官方发布](https://noitomrobotics.com/purpose-built-humanoid-data-collection-platform-for-embodied-ai/) | 正文回顾 WAIC 2025 首秀及 8 月 EAI Show 展出。诺亦腾机器人负责动捕遥操作，PNDbotics 提供本体，因时机器人提供触觉灵巧手；文章所述为固定底座配置。 |
| 2026-03 | [ModalityNet 技术博客 Episode 1](https://noitomrobotics.com/art-of-modalities/) | 提出覆盖全身运动、精细人–物交互与自然场景的三类数据。文章页面显示 2026 年 3 月，[格式规范 v1.0](https://modalitynet.com/tech-spec) 的版本表另标 2026-05-11。 |
| 2026-06 | [World Compiler 技术博客 Episode 2](https://noitomrobotics.com/world-compiler/) | 页面显示 2026 年 6 月。以高精度数据监督从第一视角视频估计深度、手部动作和缺失体段，用遮蔽后再恢复的方式核查估计误差。 |
| 2026-08-19 | [HiPHI 发布公告](https://noitomrobotics.com/noitom-robotics-releases-hiphi-one-of-the-largest-high-precision-human-motion-datasets-ever-made-public-at-the-world-robot-conference-2/) | WRC 发布数据；论文 v1 提交于 08-17。617.5 小时含左右镜像增强。见 [HiPHI 项目归档](./hiphi.md)。 |
| 2026-11-05 | [ITW-Ego Plus 公告](https://www.globenewswire.com/fr/news-release/2026/11/05/3396456/0/en/noitom-robotics-introduces-itw-ego-plus-at-corl-2026-in-the-wild-egocentric-data-with-full-body-motion.html) | 公告时间早于 11 月 9–12 日的 CoRL 展出。见 [ITW-Ego Plus 项目归档](./itw-ego-plus.md)。 |

## 开放范围

- **Adam-U：** 官方技术说明介绍 SDK 和同步数据能力。[PNDbotics 遥操作仓](../repos/pnd-teleoperation.md)提供 Noitom 动捕接入、ROS 2 重定向和预览所需的部分实现；本次查阅的产品入口未列完整数采栈源码或示范数据的下载入口。
- **ModalityNet：** [官网](https://modalitynet.com/)可浏览产品信息，完整目录需登录。本次确认 HiPHI 在 [Hugging Face](https://huggingface.co/datasets/noitomrobotics/HiPHI) 门控发布，适用 [ModalityNet Open Research License v1.0](https://modalitynet.com/open-research-license)；[本地 Viewer](../repos/hiphi.md)采用 Apache-2.0。未核实完整 HiPHI-OM 或 ITW-Ego Plus 的下载与许可。
- **联合研究：** NEAR Lab 索引列出 [OmniContact](./omnicontact-project.md)、[HiPHI](./hiphi.md)、[AdaPT](./adapt-humanoidtennis.md)和 [UMR](./umr-project.md)。时间轴使用各项目的公开事件或论文 v1 日期；论文署名与可获取资产见各项目归档。

## 对 wiki 的映射

- [诺亦腾机器人技术路线](../../wiki/entities/noitom-robotics.md)
