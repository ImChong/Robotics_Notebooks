# 万字长文｜人形机器人的运动小脑会不会成为人形机器人的基础设施？我翻完60多篇人形运控论文后的一个判断

> 来源归档（blog / 微信公众号）

- **标题：** 万字长文｜人形机器人的运动小脑会不会成为人形机器人的基础设施？我翻完60多篇人形运控论文后的一个判断
- **类型：** blog
- **作者：** 具身智能研究室（微信公众号）
- **原始链接：** https://mp.weixin.qq.com/s/Kx9myecE1Z0eGqOapoqQnA
- **发表日期：** 2026-06-17
- **入库日期：** 2026-06-18
- **抓取方式：** [Agent Reach](https://github.com/Panniantong/Agent-Reach) v1.5.0 + [wechat-article-for-ai](https://github.com/bzd6661/wechat-article-for-ai)（Camoufox）；正文约 2.9 万字 / 70 图；Jina Reader 对 `mp.weixin.qq.com` 返回 CAPTCHA，未采用
- **关联姊妹篇：** [42 篇 humanoid RL 身体系统栈](wechat_embodied_ai_lab_humanoid_rl_motion_survey.md)（`hz9JXtJeUPRfUGzfD-pZuA`）、[BFM 41 篇专题](wechat_embodied_ai_lab_bfm_41_papers_survey.md)、[Loco-Manip 8 篇周报](wechat_embodied_ai_lab_loco_manip_8_papers_survey.md)、[AMP 19 篇运动先验](wechat_embodied_ai_lab_humanoid_amp_motion_prior_survey.md)
- **一句话说明：** 以「动作小脑」重框 64 篇人形运控论文：走路是底座、全身跟踪是身体接口、Loco-Manip 是任务形态；核心判断是 VLA/世界模型与真实身体之间正在长出一层能把稀疏意图翻译成可执行、可恢复、可迁移全身运动的 **基础设施**，而非又一个单点 demo。

## 核心摘录（归纳，非全文）

### 核心判断

- **三层分工：** Locomotion = 底座；全身跟踪 = 身体接口；Loco-Manip = 边移动边干活的任务形态（当前准确度与自然度仍不足）。
- **动作小脑定义：** 不负责语言理解（VLA）或画面预测（世界模型），而负责把参考动作、遥操作、稀疏目标、任务空间命令乃至上层模糊意图 **翻译成全身可执行运动**。
- **从播放到补全：** MaskedMimic 启示——给完整 reference 是播放器；给稀疏目标还能补全全身，才像动作小脑。
- **数据炼油链：** GVHMR/TRAM → GMR/NMR/OmniRetarget → HumanX/HDMI/SUGAR/GenMimic；没有干净 reference 就没有好小脑。
- **规模化跟踪：** OmniTrack、BeyondMimic、SONIC、HoloMotion、HumanoidGPT、LIMMT、M3imic、RGMT、Any2Track 等把 motion tracking 推向 **身体基座**。
- **接口变宽：** BFM-Zero、MuGen、OMG、MotionWAM 等让 goal/reward/prompt 成为动作小脑输入。
- **真问题：** reference 脚滑、重定向穿地、抗扰、换本体、上机抖动、开门反拉等「土问题」才是基础设施瓶颈。

### 九组论文地图（A–I，64 篇）

| 组 | 主题 | 篇数 | 核心站位 |
|----|------|------|----------|
| **A** | 走路底座 | 10 | 导航/地形/跑酷/视觉穿越——没有稳定步态后面都是空中楼阁 |
| **B** | 动作模仿源流 | 5 | DeepMimic → AMP/SMP → PHC → MaskedMimic：从照着做到补着做 |
| **C** | 数据入口 | 9 | 视频恢复、重定向、交互数据生成 |
| **D** | 全身跟踪基座 | 13 | 规模化 tracking、鲁棒性、恢复与安全约束 |
| **E** | 可提示控制 | 4 | goal/reward/prompt 调用身体 |
| **F** | 跨本体与遥操作 | 5 | 身体经验复用与真机数据采集 |
| **G** | Loco-Manip 接口 | 5 | 任务空间命令、MPC 引导、上层规划分工 |
| **H** | 真实任务 | 8 | 开门、悬挂负载、重载技能、数据生成、爬梯 |
| **I** | 柔顺与接触 | 5 | 柔顺 WBC、力/触觉监督 |

## 对 wiki 的映射

- [humanoid-motion-cerebellum-technology-map](../../wiki/overview/humanoid-motion-cerebellum-technology-map.md)（本次升格 **父节点**）
- 九组分类 hub：`wiki/overview/motion-cerebellum-category-*`
- **节点复用策略：** 与 [身体系统栈 42 篇](../../wiki/overview/humanoid-rl-motion-control-body-system-stack.md) / [Loco-Manip 8 篇](../../wiki/overview/loco-manip-8-papers-technology-map.md) / [BFM 41 篇](../../wiki/overview/bfm-41-papers-technology-map.md) 重叠的论文 **复用既有 `paper-hrl-stack-*` 等实体**，仅为 15 篇尚无索引的工作新建 `paper-motion-cerebellum-*`（见 [motion_cerebellum_64_catalog.md](../papers/motion_cerebellum_64_catalog.md)）

## 可信度与使用边界

- 本文为 **第三方精读编译**；框架与 64 篇列表以 2026-06-18 Agent Reach 抓取版为准。
- 与 [身体系统栈姊妹篇](wechat_embodied_ai_lab_humanoid_rl_motion_survey.md) **论文高度重叠、组织视角不同**：系统栈按八层管线，本长文按「动作小脑」横切面（A–I）。
- 单篇细节应回到各工作 arXiv / 项目主页，不以公众号为唯一一手来源。

## 当前提炼状态

- [x] Agent Reach v1.5.0 + wechat-article-for-ai 正文抓取与归纳摘要
- [x] 父节点 + 九组分类 hub + 64 篇 catalog
- [x] 15 篇新论文 `paper-motion-cerebellum-*` 索引（其余复用既有节点）
- [x] 与身体系统栈 / BFM / Loco-Manip 姊妹篇交叉引用
