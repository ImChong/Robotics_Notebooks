# 具身智能研究室 · 人形机器人 RL 运动控制 42 篇综述（微信公众号长文）

> 来源归档（ingest）

- **标题：** 两万字长文，读懂人形机器人强化学习运动控制：42 篇论文搭起的算法圣经
- **作者：** 具身智能研究室（微信公众号）
- **类型：** site / wechat-article / survey
- **链接：** <https://mp.weixin.qq.com/s/hz9JXtJeUPRfUGzfD-pZuA>
- **入库日期：** 2026-05-18
- **发表日期：** 2026-05-18
- **抓取方式：** Camoufox（隐身 Firefox）通过 `bzd6661/wechat-article-for-ai`；正文落盘为 Markdown，约 4.5 万字 / 42 张图
- **一句话说明：** 把 42 篇 humanoid RL 运动控制 / 移动操作论文按「身体系统栈」组织（数据 → 参考 / 跟踪 → 控制 → 感知 → 接触 → 安全 → 任务接口 → 世界模型），并据此给出 6 个研究判断；核心主张：动作不是能力，**动作在真实世界精细交互闭环里**才是能力，VLA / 世界模型对身体的稳定调用是这层成熟后的结果，不是起点。

## 文章主张要点（归纳）

### 核心论点

- **动作 ≠ 能力**：同样是「翻越障碍 / 开门 / 拥抱」，参考动作能复现不代表能在真实世界里物理可执行；接触、视觉、力、负载、失败恢复都必须进入闭环。
- **判断 = 系统栈**：作者拒绝「哪篇更强」式排序，主张这批论文在共同补齐**一套从数据到世界模型的身体系统栈**——单点比拼意义有限，互补位置才是阅读关键。
- **VLA 调用是结果**：语言 / VLA 不能跳过底层控制；中间需要一层「身体 API」（skill token / latent action / 接触模式 / 短时 action chunk / 柔顺系数 / 视觉闭环目标 / 恢复策略），上层模型才能稳定调用。

### 八层身体系统栈（原文「整体判断」一节）

| 层 | 关注 | 代表论文（站位） |
|---|---|---|
| 数据层 | 人类动作 / 视频 / 遥操作 → 机器人可执行参考 | GMR, NMR, OmniRetarget, H2O, OmniH2O, HumanX, HDMI, GenMimic, TWIST / TWIST2 |
| 参考 / 跟踪控制层 | 参考动作如何进入物理仿真、稳定跟踪并在线修正 | DeepMimic, OmniTrack, BeyondMimic, Motion Generation + Tracking, Heracles |
| 控制层 | 多动作跟踪、抗扰、负载适应、恢复 | Any2Track, RGMT, OmniXtreme, SONIC, AMS, HALO |
| 感知层 | 视觉和深度如何进入动作闭环 | PHP, Deep Whole-body Parkour, Hiking in the Wild, ASAP, 视觉足球, VIRAL, DoorMan |
| 接触层 | 力、柔顺、对象动力学如何进入控制 | CHIP, GentleHumanoid, HAIC, Thor |
| 安全层 | 跌倒、异常状态、失败恢复 | SafeFall, Heracles, AMS |
| 任务接口层 | 精细身体能力如何被语言 / VLA 调用 | SENTINEL, WholeBodyVLA, OmniH2O, GR00T N1, MetaWorld, BFM-Zero |
| 世界模型层 | 动作执行前如何预测后果 / 评估策略 | Ego-Vision World Model, DreamDojo |

### 六个未来判断（原文「八、我对未来的几个判断」）

1. **动作库会继续变大，但量不是终点**——稀缺的是「带交互信息」（视觉 / 接触 / 力 / 失败恢复）的高质量数据。
2. **视觉从识别目标变成控制闭环**——sim-to-real 是核心瓶颈，身体动作和视觉输入双向影响。
3. **柔顺和力控决定数据质量**——控制器太硬 → 遥操采集质量差 → 上层模型再大也学不好。
4. **失败恢复会成为严肃指标**——论文不能只展示成功视频；冲击力、保护、异常恢复、避免伤人都要量化。
5. **VLA 调用运动控制不是起点而是结果**——必须经过「稳定移动 → 精细全身交互 → 身体 API」三步。
6. **世界模型会成为策略上线前的试运行环境**——价值在 action-conditioned rollout（预测接触后果、失败概率），不在生成好看视频。

## 对 wiki 的映射

- **新建 overview**：[`wiki/overview/humanoid-rl-motion-control-body-system-stack.md`](../../wiki/overview/humanoid-rl-motion-control-body-system-stack.md) — 提炼作者的八层身体系统栈与六个判断，并把已有 wiki 实体页（如 [DeepMimic](../../wiki/methods/deepmimic.md)、[SONIC](../../wiki/methods/sonic-motion-tracking.md)、[BeyondMimic](../../wiki/methods/beyondmimic.md)、[Any2Track](../../wiki/methods/any2track.md)、[AMS](../../wiki/methods/ams.md)、[GMR](../../wiki/methods/motion-retargeting-gmr.md)、[NMR](../../wiki/methods/neural-motion-retargeting-nmr.md)、[DoorMan 论文](../../wiki/entities/paper-doorman-opening-sim2real-door.md)、[VIRAL 论文](../../wiki/entities/paper-viral-humanoid-visual-sim2real.md)、[BFM 论文](../../wiki/entities/paper-behavior-foundation-model-humanoid.md)、[GR00T-WholeBodyControl](../../wiki/entities/gr00t-wholebodycontrol.md)、[ULTRA Survey](../../wiki/tasks/ultra-survey.md)）按层挂接。

## 备注（维护者）

- 微信链接对 Jina Reader / 多数 headless 客户端会返回 CAPTCHA 墙；本次抓取走 Camoufox + 自签 CA 信任，落盘后只引用、不在 wiki 转存正文。
- 文中提及的若干工作（OmniRetarget, OmniTrack, RGMT, OmniXtreme, HALO, Thor, SENTINEL, WholeBodyVLA, MetaWorld, SafeFall, GenMimic, HumanX, HDMI, Hiking in the Wild, PHP, Ego-Vision World Model, DreamDojo, GentleHumanoid, HAIC, CHIP, TWIST, TWIST2, PvP, Adaptive Humanoid Control, ASAP）暂未在本仓库建立独立 entity / method 页；后续如要单篇升格，应回到该论文官方主页 / arXiv 另起 sources 条目，不直接以本公众号为唯一来源。
- 公众号文本可能随时被作者编辑或删除；本归档以 2026-05-18 抓取版为准，引用以原文作者表述为准，本仓库只做框架级转述。
