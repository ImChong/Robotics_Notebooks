# 两万字长文，读懂人形机器人强化学习运动控制：42 篇论文搭起的算法圣经

> 来源归档（blog / 微信公众号）

- **标题：** 两万字长文，读懂人形机器人强化学习运动控制：42 篇论文搭起的算法圣经
- **类型：** blog
- **作者：** 具身智能研究室（微信公众号）
- **原始链接：** https://mp.weixin.qq.com/s/hz9JXtJeUPRfUGzfD-pZuA
- **入库日期：** 2026-05-21
- **抓取方式：** [Agent Reach](https://github.com/Panniantong/Agent-Reach) v1.4.0 安装的 `wechat-article-for-ai`（Camoufox）；正文约 4.7 万字 / 42 图；Jina Reader 对该链接触发微信 CAPTCHA，未采用
- **关联姊妹篇：** [万字长文，读懂人形机器人 AMP：20 篇论文搭起的运动先验圣经](wechat_embodied_ai_lab_humanoid_amp_motion_prior_survey.md)（`https://mp.weixin.qq.com/s/YZsm3855iP3TNTTt1aou7w`）
- **一句话说明：** 把 42 篇 humanoid RL 运动控制 / 移动操作论文按「身体系统栈」组织（数据 → 参考 / 跟踪 → 控制 → 感知 → 接触 → 安全 → 任务接口 → 世界模型），并给出 6 个研究判断；核心主张：动作不是能力，**动作在真实世界精细交互闭环里**才是能力，VLA / 世界模型对身体的稳定调用是这层成熟后的结果，不是起点。

## 核心摘录（归纳，非全文）

### 核心论点

- **动作 ≠ 能力**：参考动作能复现 ≠ 真实世界物理可执行；接触、视觉、力、负载、失败恢复须进入闭环。
- **判断 = 系统栈**：单点「哪篇更强」意义有限；这批论文在共同补齐**从数据到世界模型的身体系统栈**。
- **VLA 调用是结果**：语言 / VLA 不能跳过底层控制；需「身体 API」后上层才能稳定调用。

### 八层身体系统栈（原文「整体判断」）

| 层 | 关注 | 代表论文（站位） |
|---|---|---|
| 数据层 | 人类动作 / 视频 / 遥操作 → 机器人可执行参考 | GMR, NMR, OmniRetarget, H2O, OmniH2O, HumanX, HDMI, GenMimic, TWIST / TWIST2 |
| 参考 / 跟踪控制层 | 参考动作进入物理仿真、稳定跟踪、在线修正 | DeepMimic, OmniTrack, BeyondMimic, Motion Generation + Tracking, Heracles |
| 控制层 | 多动作跟踪、抗扰、负载适应、恢复 | Any2Track, RGMT, OmniXtreme, SONIC, AMS, HALO |
| 感知层 | 视觉和深度进入动作闭环 | PHP, Deep Whole-body Parkour, Hiking in the Wild, ASAP, 视觉足球, VIRAL, DoorMan |
| 接触层 | 力、柔顺、对象动力学进入控制 | CHIP, GentleHumanoid, HAIC, Thor |
| 安全层 | 跌倒、异常状态、失败恢复 | SafeFall, Heracles, AMS |
| 任务接口层 | 精细身体能力被语言 / VLA 调用 | SENTINEL, WholeBodyVLA, OmniH2O, GR00T N1, MetaWorld, BFM-Zero |
| 世界模型层 | 动作执行前预测后果 / 评估策略 | Ego-Vision World Model, DreamDojo |

### 六个未来判断（原文「八、我对未来的几个判断」）

1. 动作库会继续变大，但稀缺的是「带交互信息」的高质量数据。
2. 视觉从识别目标变成控制闭环；sim-to-real 是核心瓶颈。
3. 柔顺和力控决定数据质量；控制器太硬 → 遥操与上层学习都受损。
4. 失败恢复会成为严肃指标。
5. VLA 调用运动控制不是起点而是结果。
6. 世界模型会成为策略上线前的试运行环境（action-conditioned rollout，非好看视频）。

## 对 wiki 的映射

- [humanoid-rl-motion-control-body-system-stack](../../wiki/overview/humanoid-rl-motion-control-body-system-stack.md)（升格主页面）
- [humanoid-amp-motion-prior-survey](../../wiki/overview/humanoid-amp-motion-prior-survey.md)（姊妹篇 AMP 运动先验综述）
- [robot-world-models-training-loop-taxonomy](../../wiki/overview/robot-world-models-training-loop-taxonomy.md)

## 可信度与使用边界

- 本文为 **第三方精读编译**；框架与论文列表以 2026-05-21 Agent Reach 抓取版为准。
- 与 [`sources/sites/wechat-embodied-ai-lab-humanoid-rl-motion-survey-2026-05-18.md`](../sites/wechat-embodied-ai-lab-humanoid-rl-motion-survey-2026-05-18.md) 为同一公众号长文；本 `blogs/` 条目对齐 Agent Reach 抓取流程，站点归档保留首次入库记录。
- 单篇论文细节应回到各工作 arXiv / 项目主页，不以公众号为唯一一手来源。

## 当前提炼状态

- [x] Agent Reach + Camoufox 正文抓取与归纳摘要
- [x] wiki 主页面映射确认
- [x] 与 AMP 姊妹篇交叉引用
