---
type: entity
project_id: dreamlabs-dream-1
project: https://dreamlabs.ai/research/dream-1
tags: [robot-learning, wam, manipulation, api, dream-labs]
status: complete
updated: 2026-10-20
related:
  - ../concepts/world-action-models.md
  - ./paper-notebook-dreamzero-world-action-models-are-zero-shot-poli.md
  - ../methods/vla.md
sources:
  - ../../sources/sites/dream-labs-dream-1.md
summary: "Dream-1 是 Dream Labs 面向双臂平行夹爪机器人的 World Action Model API，官方称可零样本尝试未见机器人、任务与环境；目前以研究 beta 公共 API 提供，未列出开源代码或权重。"
---

# Dream-1

**Dream-1 是 Dream Labs 提供的 World Action Model API，目标是让双臂平行夹爪机器人通过公共 API 尝试执行未见任务，而无需先为新机器人单独训练模型。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| WAM | World Action Model | Dream-1 所属的世界预测与动作生成模型路线 |
| API | Application Programming Interface | 用户通过网络客户端提交观测并请求机器人动作的服务接口 |
| IK | Inverse Kinematics | 将末端目标转换为关节运动的逆运动学求解器，影响跨本体控制效果 |

## 核心信息

| 字段 | 内容 |
|------|------|
| 机构 | 梦想实验室（Dream Labs） |
| 项目主页 | <https://dreamlabs.ai/research/dream-1> |
| 服务形式 | 公共 API（research beta） |
| 官方开源代码 / 权重 | 项目页未列出 |

## 为什么重要

Dream-1 把 WAM 从研究模型推进到可调用的机器人服务：开发者可以直接接入支持的机器人，在 API 中选择模型并请求动作。Dream Labs 将其定位为解决“新机器人需要专属数据、专属策略”的数据冷启动问题。

Dream Labs 的项目页把 DreamZero 研究列为团队的起点：其早期实验发现，少量 YAM 机器人数据后训练可迁移到未见任务；Dream-1 则是团队目前公开提供的 WAM API 产品形态。两者相关，但 Dream-1 的页面没有披露它与 DreamZero 检查点之间的具体继承关系。

## 核心机制与接口

官方披露的使用流程是：创建客户端并配置 API key，选择 `dreamlabs/dream-1` checkpoint，提交当前观测，再调用 `get_action()` 取得动作。服务目标是将机器人控制模型作为 API 提供，降低使用者自行训练和部署模型的门槛。

项目页没有公开 Dream-1 的模型结构、训练数据组成、动作表示或服务端推理细节，因此不应仅根据 WAM 标签推断其采用了 DreamZero 的特定骨干或训练配方。

## 官方零样本评测

官方页面给出 27 种相对 YAM 未见机器人的仿真结果，并以 YAM 表现归一化为 100%。每种机器人 300 次 rollout：150 次用于指令跟随评估，150 次用于未见任务泛化评估。下表是官方相对分数的部分示例，不是任务成功率。

| 机器人 | 相对 YAM 分数 |
|---|---:|
| PiPER（双臂） | 91.2% |
| Trossen Workbench | 82.8% |
| PR2 | 73.8% |
| Unitree R1 | 63.9% |
| SO-101 | 54.2% |
| Mantis | 45.8% |

这组分数说明迁移能力因机器人本体而变化；不能将相对分数直接解释成真实环境的绝对成功率。官方页面还展示多地点测试案例，但这些示例不能替代统一的实机基准。

## 工程实践

1. 确认目标机器人属于双臂平行夹爪使用范围，并准备可供客户端读取的观测。
2. 按官方 Getting Started 指南配置客户端、API key 和 `dreamlabs/dream-1` checkpoint。
3. 检查本体 IK 求解器与动作映射；官方指出零样本性能部分取决于 IK。
4. 先在受控、低风险环境验证动作质量，再逐步扩大任务范围。
5. 如果基础模型未达到目标任务需求，官方页面建议收集本体数据并使用其 FaaS 后训练服务。

## 局限与风险

- **研究 beta：** 官方明确说明模型会犯错，且未验证用于安全关键场景；机器人应在受控环境测试，特别注意人员周边运行风险。
- **支持范围有限：** 当前定位是带平行夹爪的双臂操作；页面未证明其支持灵巧手、人形全身控制或移动操作。
- **结果依赖本体：** 仿真相对分数跨机器人波动明显，且官方指出 IK 会影响迁移表现；实机效果需要按具体硬件单独验证。
- **开源边界：** 项目页提供公共 API 和客户端示例，但没有列出可下载代码仓库、权重或数据集，因此目前属于可调用服务，而非可本地复现的开源模型。
- **价格：** 官方页面标示服务起价为 $3/小时；实际费用与可用额度应以服务端为准。

## 关联页面

- [World Action Models（WAM）](../concepts/world-action-models.md)：理解 Dream-1 所属模型范式及与 VLA、世界模型的区别。
- [DreamZero](./paper-notebook-dreamzero-world-action-models-are-zero-shot-poli.md)：Dream Labs 在项目页中提及的早期研究背景。
- [Vision-Language-Action（VLA）](../methods/vla.md)：对照具身策略常见的视觉、语言与动作建模路线。

## 参考来源

- [Dream-1 官方项目页归档](../../sources/sites/dream-labs-dream-1.md)
- [Dream-1 官方研究页](https://dreamlabs.ai/research/dream-1)
- [DreamZero 深读摘要](./paper-notebook-dreamzero-world-action-models-are-zero-shot-poli.md)

## 推荐继续阅读

- [Dream-1: Zero-Shot to New Robots](https://dreamlabs.ai/research/dream-1)
- [DreamZero: World Action Models are Zero-shot Policies](https://arxiv.org/abs/2602.15922)