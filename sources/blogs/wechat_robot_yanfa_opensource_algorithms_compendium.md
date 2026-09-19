# 机器人研发开源机器人算法大全

> 来源归档（blog / 微信公众号）

- **标题：** 机器人研发开源机器人算法大全
- **类型：** blog / curated-index
- **作者：** 机器人研发工程师（微信公众号）
- **原始链接：** https://mp.weixin.qq.com/s/eXDsk8svhmLUkzMVlXd-qw
- **入库日期：** 2026-09-19
- **抓取方式：** WebFetch（原文 HTML）；`wechat-article-for-ai` 环境缺 `markdownify` 时未用 Camoufox 重抓
- **原始抓取落盘：** [`sources/raw/wechat_robot_yanfa_opensource_algorithms_2026.md`](../raw/wechat_robot_yanfa_opensource_algorithms_2026.md)
- **一句话说明：** 面向机器人研发工程师的 16 项开源算法/框架/索引清单，覆盖 RL 步态、模仿学习、VLA、动力学库、MuJoCo 资产与 Awesome 导航；文内部分 GitHub 链接已失效，入库时已逐步 2.5 核查并校正。
- **沉淀到 wiki：** [`wiki/overview/robot-opensource-algorithms-compendium-wechat.md`](../../wiki/overview/robot-opensource-algorithms-compendium-wechat.md)

## 步骤 2.5：链接与开源核查（2026-09-19）

| # | 文内名称 | 原文 URL | 核查结论 | 校正后可用 URL |
|---|----------|----------|----------|----------------|
| 1 | rsl-rl | leggedrobotics/rsl_rl | **已开源** BSD-3-Clause | 同左 ✓ |
| 2 | AMP | facebookresearch/amp | **404**；Meta 未托管同名仓 | 论文节点 + [escontra/AMP_for_hardware](https://github.com/escontra/AMP_for_hardware) |
| 3 | TienKung-Lab | Open-X-Humanoid/TienKung-Lab | **已开源** | 同左 ✓ |
| 4 | unitree_rl_gym | unitreerobotics/unitree_rl_gym | **已开源** | 同左 ✓ |
| 5 | ALOHA-ACT | StanfordVL/ALOHA | **404** | [tonyzhaozh/aloha](https://github.com/tonyzhaozh/aloha) + [tonyzhaozh/act](https://github.com/tonyzhaozh/act) |
| 6 | Diffusion-Policy | real-strawberry/diffusion_policy | **404**（org 笔误） | [real-stanford/diffusion_policy](https://github.com/real-stanford/diffusion_policy) ✓ |
| 7 | OpenVLA | openvla/openvla | **已开源** Apache-2.0 | 同左 ✓ |
| 8 | Octo | octo-models/octo | **已开源** | 同左 ✓ |
| 9 | LingBot | antgroup/lingbot | **404** | [robbyant/lingbot-world](https://github.com/robbyant/lingbot-world)（世界模型主线） |
| 10 | LeRobot | huggingface/lerobot | **已开源** Apache-2.0 | 同左 ✓ |
| 11 | Pinocchio | stack-of-tasks/pinocchio | **已开源** BSD-2-Clause | 同左 ✓ |
| 12 | Acados | acados/acados | **已开源** BSD-2-Clause | 同左 ✓ |
| 13 | mujoco_menagerie | google-deepmind/mujoco_menagerie | **已开源** Apache-2.0 | 同左 ✓ |
| 14 | unitree_mujoco | unitreerobotics/unitree_mujoco | **已开源** | 同左 ✓ |
| 15 | awesome-legged-locomotion-learning | gaiyi7788/... | **已开源** 列表 | 同左 ✓ |
| 16 | awesome-physical-ai | natnew/... | **已开源** MIT | 同左 ✓ |

## 对 wiki 的映射

- [robot-opensource-algorithms-compendium-wechat](../../wiki/overview/robot-opensource-algorithms-compendium-wechat.md) — 16 项独立详情节点索引
