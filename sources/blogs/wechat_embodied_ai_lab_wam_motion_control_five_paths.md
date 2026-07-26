# 关于人形机器人运动控制怎样进入世界模型？我的一些判断和思考——以及它会在 WAM 的在线规划、动力学建模和真机执行中处于什么位置

> 来源归档（blog / 微信公众号）

- **标题：** 关于人形机器人运动控制怎样进入世界模型？我的一些判断和思考——以及它会在 WAM 的在线规划、动力学建模和真机执行中处于什么位置
- **类型：** blog
- **作者：** 具身智能研究室（微信公众号）
- **原始链接：** https://mp.weixin.qq.com/s/2pP9LWlsTmTAgTglFuLwSA
- **发表日期：** 2026-07-26
- **入库日期：** 2026-07-26
- **抓取方式：** [Agent Reach](https://github.com/Panniantong/Agent-Reach) v1.5.0 + `wechat-article-for-ai`（Camoufox）；`--no-images`；Jina Reader 对 `mp.weixin.qq.com` 返回 CAPTCHA，未采用
- **一句话说明：** 判断 **WAM 进入物理世界时运动控制会逐渐成为其一部分**；按 **五种系统位置** 串读 10 篇工作：在线规划（Ego-VCP）、模型式 RL/动力学适配（RWM-U、LIFT）、内部动力学估计（HAIC）、带未来预测的动作策略（MotionWAM、Being-M0.7、Being-H0.7）、策略评估与动作表示（1XWM、EgoWM、UniT）。

## 核心摘录（归纳，非全文）

### 总判断

- **WAM 真要进入物理世界，运动控制会逐渐成为它的一部分。**
- 企业端：运动控制可能成为 WAM 项目中把「模型生成的目标/轨迹/动作」变成「真机可稳定执行的全身控制」的环节。
- 保守倾向：**短时域潜在动力学、隐藏状态估计、训练阶段世界模型** 会先进入人形运动控制栈；视频 WM / 世界动作策略空间更大、验证更难。

### 五种连接位置（文内 taxonomy）

| 位置 | 代表工作 | 世界模型在回路中的角色 |
|------|----------|------------------------|
| **① 在线规划** | Ego-VCP | 潜在空间展开候选动作序列，价值/失败概率筛选后只执行第一步；重规划兜底 |
| **② 模型式 RL / 动力学适配** | RWM-U、LIFT | 想象轨迹上继续训策略；不确定性惩罚或解析动力学先验约束误差累积 |
| **③ 内部动力学估计** | HAIC | 观测与策略之间的隐藏状态估计（物体相对运动），**不**替策略搜索动作 |
| **④ 带未来预测的动作策略** | MotionWAM、Being-M0.7、Being-H0.7 | 未来预测压进训练目标/潜特征；部署更接近动作策略，底层仍靠跟踪器/AMO |
| **⑤ 策略评估与动作表示** | 1XWM、EgoWM、UniT | 控制回路外侧：评测排序、动作条件视频预测、跨本体运动分词 |

### 文内关键数字与入口（策展口径，以原论文/项目页为准）

| 工作 | 亮点（公众号口径） | 入口 |
|------|-------------------|------|
| Ego-VCP | G1 上高层规划 **25 Hz**、每轮 **1024** 候选、时域 **4** 步；低层 RL 控制器跟全身 | <https://ego-vcp.github.io/> |
| RWM-U | 离线真机数据学转移 + 集成不确定性；Franka / ANYmal-D / G1 | <https://arxiv.org/abs/2504.16680> |
| LIFT | SAC 大规模预训练 + 拉格朗日先验世界模型；真机确定性、探索留在模型 | <https://lift-humanoid.github.io/> |
| HAIC | 本体历史 → 物体状态/速度/加速度 → 几何先验投影 → 学生策略 | <https://haic-humanoid.github.io/> |
| MotionWAM | 视频 DiT 中间特征 → 统一全身 motion token；G1 9 类真机任务；底层仍是 SONIC | <https://arxiv.org/abs/2606.09215> |
| Being-M0.7 | 低频潜在 video-motion 计划 + 高频动作专家 | <https://research.beingbeyond.com/being-m07> |
| Being-H0.7 | 先验/后验双分支潜空间对齐；上身 Being-H0.7、下身+腰 AMO | <https://research.beingbeyond.com/being-h07> |
| 1XWM | 动作条件未来观测 + 任务成败预测，用于策略/checkpoint 排序 | <https://www.1x.tech/discover/redwood-ai-world-model> |
| EgoWM | 预训练视频扩散 + 轻量动作条件；3-DoF 移动到 25-DoF 人形 | <https://egowm.github.io/> |
| UniT | 人–人形共享运动分词器；策略条件与 WM 动作条件入口 | <https://xpeng-robotics.github.io/unit/> |

### 作者强调的四个开放问题

1. **动作差一点，预测会不会真的不同**（动作忠实度 vs 画面连贯）
2. **预测错了谁来兜底**（重规划 / 底层跟踪 / 安全层）
3. **模型知不知道自己没见过**（不确定性进反馈）
4. **换本体后还能留下什么**（共享表示 vs 执行器动力学）

## 对 wiki 的映射

- 主升格：[wam-motion-control-five-paths](../../wiki/overview/wam-motion-control-five-paths.md)
- 论文实体（复用已有，不新建重复节点）：
  - [Ego-VCP / Ego-Vision WM](../../wiki/entities/paper-hrl-stack-33-ego_vision_world_model_for_humanoid.md)
  - [RWM-U](../../wiki/entities/robotic-world-model-eth-rsl.md)
  - [LIFT](../../wiki/entities/lift-humanoid.md)
  - [HAIC](../../wiki/entities/paper-haic.md)
  - [MotionWAM](../../wiki/entities/paper-motionwam-humanoid-loco-manipulation-wam.md)
  - [Being-M0.7](../../wiki/entities/paper-being-m07-humanoid-latent-wam.md)
  - [Being-H0.7](../../wiki/methods/being-h07.md)
  - [1XWM](../../wiki/entities/paper-1xwm-redwood-world-model.md)（新建）
  - [EgoWM](../../wiki/entities/paper-egowm-egocentric-world-model.md)（新建）
  - [UniT](../../wiki/entities/paper-unit-unified-physical-language.md)（新建）
- 相关概念：[World Action Models](../../wiki/concepts/world-action-models.md)、[Generative World Models](../../wiki/methods/generative-world-models.md)、[robot-world-models-training-loop-taxonomy](../../wiki/overview/robot-world-models-training-loop-taxonomy.md)

## 可信度与使用边界

- 本文为 **微信公众号策展判断文**，五路径 taxonomy 与系统位置读法以本文为准；方法数字、开源状态与评测以各论文 PDF / 项目页为准。
- 不把公众号作为唯一一手来源；每篇论文实体页的「参考来源」同时挂 arXiv / 项目页 / 代码归档。

## 当前提炼状态

- [x] 正文抓取与五种位置归纳
- [x] 10 篇入口与 wiki 映射（含去重）
- [x] 升格 overview + 补齐缺失论文实体
