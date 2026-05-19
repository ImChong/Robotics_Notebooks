# 语言控制人形机器人，真正缺的是语义到身体的接口

> 来源归档（blog / 微信公众号）

- **标题：** 语言控制人形机器人，真正缺的是语义到身体的接口
- **类型：** blog
- **作者：** 具身智能研究室（微信公众号）
- **原始链接：** https://mp.weixin.qq.com/s/u1ZUaFGYRKXxMcS7-V_2WA
- **入库日期：** 2026-05-19
- **抓取方式：** Agent Reach `wechat-article-for-ai`（Camoufox）
- **编译所依论文：** [DAJI（arXiv:2605.14417）](../papers/daji_arxiv_2605_14417.md) — *Before the Body Moves: Learning Anticipatory Joint Intent for Language-Conditioned Humanoid Control*
- **一句话说明：** 策展解读 DAJI：语言控制人形的稀缺层不是「听懂」，而是 **语义–动力学之间的中间接口**；预期关节意图在流式指令与闭环控制中携带 **支撑切换、接触与平衡准备**，消融显示去掉未来信息后长程成功率崩塌。

## 核心摘录（归纳，非全文）

### 1) 主判断

- 大模型接人形后，「一句话动起来」的想象忽略了：**语言指令还要经过平衡、支撑切换、接触与动作前身体准备**。
- **真正稀缺的是语义到身体的接口**：既要表达意图，又要能被真实身体 **连续执行**；接口不成立时，上层理解再强也会在重心/支撑/接触时机上失败。

### 2) 参考轨迹路线的局限

- 上层生成运动参考、底层追踪很模块化，但在真实人形上参考常 **未考虑当前重心、接触、动量与可行延续** → 底层只能被动修（修多不像，修少不稳）。
- DAJI 不把参考轨迹当最终接口，而改用 **可执行、带预期信息的关节意图**。

### 3) 与 OmniH2O / 表示谱系

- **OmniH2O**：VR/RGB/语言等多输入统一到 **可部署身体接口**；DAJI 输入侧更偏 **语言流式生成**，底层问题类似：**上层输入 ≠ 电机动作**。
- **MDM / ASE / CALM**：语言或技能表示驱动人体运动；DAJI 更近 **低延迟闭环人形控制**。

### 4) DAJI 机制（DAJI-Flow + DAJI-Act）

- **预期关节意图**：低频中间表示，带 **未来短时间身体准备**；Flow 生成意图片段，Act 结合本体感知解码高频动作。
- **「预期」**：转身前准备重心、迈步前切换支撑——轨迹描述「想要什么」，意图还描述 **为接住下一步现在应准备什么**。

### 5) 实验要点（编译稿数字，以论文为准）

- 部署：64 维 MLP、CPU 约 4.71 ms、成功率约 80.8%；16 维约 49.5%。
- HumanML3D 风格：DAJI 成功率约 **94.42%** vs TextOp 90.00%、MotionStreamer 87.50%。
- BABEL 流式：子序列 FID **0.152** vs TextOp **0.538**。
- **消融**：去掉未来信息 → 60s 成功率 **87% → 10%**（强调身体无重置键、片段间状态必须连续）。

### 6) 路线判断

- 未来争夺点：**语义到身体的中间接口**（关节意图、skill token、action chunk、接触模式等），须 **既能接住语言，又能接住身体**。

## 对 wiki 的映射

- [paper-daji-anticipatory-joint-intent](../../wiki/entities/paper-daji-anticipatory-joint-intent.md)（本次升格主页面）
- [vla](../../wiki/methods/vla.md)、[loco-manipulation](../../wiki/tasks/loco-manipulation.md)、[humanoid-rl-motion-control-body-system-stack](../../wiki/overview/humanoid-rl-motion-control-body-system-stack.md)（任务接口层）

## 可信度与使用边界

- 本文为 **第三方精读**；方法细节、表格数字与消融以 [arXiv:2605.14417](https://arxiv.org/abs/2605.14417) 与 [项目页](../sites/daji-hxxxz0-github-io.md) 为准。

## 当前提炼状态

- [x] 正文抓取与归纳
- [x] 一手论文索引
- [x] wiki 实体页映射确认
