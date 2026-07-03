# 我把最近 Loco-Manip 工作重新梳理了一遍：人形机器人怎样与物理世界接触，数据、策略、力控和 VLA 各自解决什么

> 来源归档（blog / 微信公众号）

- **标题：** 我把最近 Loco-Manip 工作重新梳理了一遍：人形机器人怎样与物理世界接触，数据、策略、力控和 VLA 各自解决什么
- **类型：** blog
- **作者：** 具身智能研究室（微信公众号）
- **原始链接：** https://mp.weixin.qq.com/s/UjShbwl8p1h9ukymfiRNaw
- **发表日期：** 2026-07-03
- **入库日期：** 2026-07-03
- **抓取方式：** [Agent Reach](https://github.com/Panniantong/Agent-Reach) v1.5.0（`pip install git+https://github.com/Panniantong/Agent-Reach.git` + [wechat-article-for-ai](https://github.com/bzd6661/wechat-article-for-ai) 至 `~/.agent-reach/tools/`（Camoufox，`playwright==1.49.1` 规避 viewport 协议错误））；正文约 0.77 万字 / 7 图；Jina Reader 对 `mp.weixin.qq.com` 返回 CAPTCHA，未采用
- **姊妹篇：** [161 篇人形 loco-manip 十类地图](wechat_embodied_ai_lab_humanoid_loco_manip_161_survey.md)、[Loco-Manip 8 篇数据入口](wechat_embodied_ai_lab_loco_manip_8_papers_survey.md)、[运动小脑 64 篇](wechat_embodied_ai_lab_humanoid_motion_cerebellum_survey.md)
- **一句话说明：** 按 **接触数据 → 接触表示/接口 → 生成式补数 → 接触后力控/柔顺 → VLA/世界模型调用** 五段链路串读约 36 篇 loco-manip 工作；核心判断：人形 loco-manip 的瓶颈正从「会不会动」转向 **接触结构能否贯穿数据、策略、力反馈与上层模型**。

## 核心摘录（归纳，非全文）

### 问题重框

- **接触 ≠ 手碰到物体**：含脚底支撑、重心、物体受力、负载摆动、触觉与上层全身动作调度。
- **策展主线：** 接触数据怎么来？接触怎么进入策略？生成式方法怎么补长尾？接触发生后怎么稳住？VLA/WM 怎么调用接触能力？

### 五段链路（对应 5 组子节点）

| 段 | 核心问题 | 代表工作 |
|----|----------|----------|
| **01 接触数据** | 带物体状态、场景约束、接触时序、本体可执行性的交互数据从哪来？ | OmniRetarget、HumanX、HDMI、SUGAR、HumanoidMimicGen、VLK 等 |
| **02 接触表示** | 接触被写成什么信号进入策略？ | SceneBot、OmniContact、CoorDex、HALOMI、WT-UMI、CEER、Pro-HOI、CWI |
| **03 生成式补数** | 生成视频/资产/仿真能否提供可训练接触轨迹？ | GenHOI、Imagine2Real、GRAIL、Humanoid-DART、LEGS、OASIS、SIMPLE |
| **04 接触后稳定** | 接触建立后力、柔顺、负载与强接触下身体如何持续？ | FALCON、HMC、WoCoCo、GentleHumanoid、CHIP、HOIST、Thor |
| **05 VLA/WM 调用** | 上层模型能否调用带接触结构的全身动作接口？ | OpenHLM、WholeBodyVLA、ROVE、MotionWAM、HAIC、WOLF-VLA |

### 开放问题（文内收束）

统一接触接口、生成式物理一致性、触觉力数据规模化、VLA 接触因果、各层接口互通——五类问题尚未收敛。

## 对 wiki 的映射

- [loco-manip-contact-technology-map](../../wiki/overview/loco-manip-contact-technology-map.md)（**父节点** + Mermaid）
- 子分类 hub：`wiki/overview/loco-manip-contact-category-01-contact-data.md` … `05-vla-world-models.md`
- **论文实体**：文中 **36 篇** 均在 `wiki/entities/paper-*` 有独立节点，并由五组 `loco-manip-contact-category-*` 分类 hub 挂接（含新建 Human-as-Humanoid、HumanoidUMI、VLK、Imagine2Real、Humanoid-DART、WOLF-VLA）

## 可信度与使用边界

- 本文为 **微信公众号策展导读**，论文细节以 arXiv / 项目页为准。
- 与 [161 篇地图](../../wiki/overview/humanoid-loco-manip-161-papers-technology-map.md) **交叉覆盖、视角不同**：161 篇按能力形成十类；本页按 **接触横切面** 五段链路。
- 原始抓取正文见 [wechat_loco_manip_contact_2026-07-03.md](../raw/wechat_loco_manip_contact_2026-07-03/我把最近Loco-Manip工作重新梳理了一遍：人形机器人怎样与物理世界接触，数据、策略、力控和VLA各自解决什么.md)。
