# 世界模型相关工作突然密集出现：机器人开始从理解图像走向预测动作如何改变物理世界

> 来源归档（blog / 微信公众号）

- **标题：** 世界模型相关工作突然密集出现：机器人开始从理解图像走向预测动作如何改变物理世界
- **类型：** blog
- **作者：** 具身智能研究室（微信公众号）
- **原始链接：** https://mp.weixin.qq.com/s/a5ZDDv70CLDfY98mfviWuA
- **发表日期：** 2026-07-11
- **入库日期：** 2026-07-11
- **抓取方式：** Agent Reach v1.5.0 + `wechat-article-for-ai`（Camoufox）；`--no-images`
- **姊妹篇：** [机器人世界模型训练闭环 taxonomy](wechat_embodied_ai_lab_robot_world_model_training_loop.md)、[Loco-Manip 接触五段链路](wechat_embodied_ai_lab_loco_manip_contact_survey.md)
- **一句话说明：** 按 **动作后果预测 / 接触状态建模 / 3D·4D 几何 / 训练与评估闭环** 四条线串读 12 篇近期世界模型工作；核心判断：机器人正从「理解当前画面」走向 **在动作执行前预测物理世界如何演化**，WAM 已分化出 **直接执行、在线修正、部署筛选** 三类接口。

## 核心摘录（归纳，非全文）

### 问题重框

- 短任务里「抓起来就结束」可遮住误差；长任务中物体滑动、形变、目标位移会使前一步偏差放大。
- 世界模型近期变化：**动作发出去之前，多一次与物理后果有关的判断**。

### 四条策展主线

| 段 | 代表工作 | 核心问题 |
|----|----------|----------|
| **01 动作后果预测** | DSWAM、DynaWM、DreamSteer、Worldscape-MoE | WAM 如何直接执行、修正基础 VLA 或部署前筛选候选动作？ |
| **02 接触状态建模** | VT-WAM、TACO、Current as Touch、Deform360 | 触觉/电流/形变如何进入状态转移与纠错数据？ |
| **03 3D/4D 几何** | RynnWorld-4D、MECo-WAM、EmbodiedGen V2 | RGB 之外如何补深度、光流、形变与可仿真环境层？ |
| **04 训练与评估闭环** | GigaWorld-1（+ DREAMSTEER/TACO/EmbodiedGen 交叉） | 世界模型能否可靠评估策略、扩展失败恢复与环境？ |

### 文内关键数字（策展口径，以论文为准）

| 工作 | 报告亮点 |
|------|----------|
| DSWAM | 真机折叠成功率 92.5%→96.3%，用时 2:18→1:44 |
| DynaWM | 冻结基础 VLA + 在线流匹配重生成动作块（移动目标） |
| DreamSteer | 四组真机基准成功率 23.75%→66.25%，指令遵循 38.75%→56.25% |
| VT-WAM | 六类接触任务平均成功率 71.67%（+26.67pp vs Fast-WAM） |
| TACO | 相对基础策略 +44pp 绝对成功率 |
| EmbodiedGen V2 | 83.3% 任务世界免改可用；RL 仿真 9.7%→79.8%，真机 21.7%→75.0% |

### 开放问题（文内收束）

动作忠实度、长时序误差累积、不确定性表达、跨本体动作接口——架构尚未收敛。

## 对 wiki 的映射

- [robot-world-models-action-consequence-technology-map](../../wiki/overview/robot-world-models-action-consequence-technology-map.md)（**父节点**）
- 子分类 hub：`wiki/overview/wm-action-consequence-category-01-wam-action-prediction.md` … `04-eval-posttrain.md`
- **论文实体（12 篇，各独立节点）**：
  - [paper-dswam-dual-system-wam](../../wiki/entities/paper-dswam-dual-system-wam.md)
  - [paper-dynawm-vla-online-correction](../../wiki/entities/paper-dynawm-vla-online-correction.md)
  - [paper-dreamsteer-vla-deployment-steering](../../wiki/entities/paper-dreamsteer-vla-deployment-steering.md)
  - [paper-worldscape-moe-heterogeneous-action](../../wiki/entities/paper-worldscape-moe-heterogeneous-action.md)
  - [paper-vt-wam-visuotactile-contact-rich](../../wiki/entities/paper-vt-wam-visuotactile-contact-rich.md)
  - [paper-taco-tactile-wm-vla-posttrain](../../wiki/entities/paper-taco-tactile-wm-vla-posttrain.md)
  - [paper-current-as-touch-proprioceptive-contact](../../wiki/entities/paper-current-as-touch-proprioceptive-contact.md)
  - [paper-deform360-deformable-visuotactile-dataset](../../wiki/entities/paper-deform360-deformable-visuotactile-dataset.md)
  - [paper-rynnworld-4d-rgb-depth-flow](../../wiki/entities/paper-rynnworld-4d-rgb-depth-flow.md)
  - [paper-meco-wam-4d-geometry-cotraining](../../wiki/entities/paper-meco-wam-4d-geometry-cotraining.md)
  - [paper-embodiedgen-v2-sim-ready-world-engine](../../wiki/entities/paper-embodiedgen-v2-sim-ready-world-engine.md)
  - [paper-gigaworld-1-policy-evaluation](../../wiki/entities/paper-gigaworld-1-policy-evaluation.md)

## 可信度与使用边界

- 本文为 **微信公众号策展导读**；论文细节以 arXiv / 项目页为准。
- 与 [robot-world-models-training-loop-taxonomy](../../wiki/overview/robot-world-models-training-loop-taxonomy.md) **互补**：该页按综述三线 taxonomy；本页按 **2026-07 动作后果横切面** 读近期 WAM 密集工作。
- 原始抓取正文见 [sources/raw/wechat_robot_world_models_action_consequence_2026-07-11/](../raw/wechat_robot_world_models_action_consequence_2026-07-11/世界模型相关工作突然密集出现：机器人开始从理解图像走向预测动作如何改变物理世界.md)。

## 当前提炼状态

- [x] 正文抓取与归纳摘要
- [x] 12 篇论文独立 wiki 节点
- [x] 四条分类 hub + 父技术地图
