# 都是机器人控制，为什么「走路」用 RL，「干活」却用 Transformer？

> 来源归档（blog / 微信公众号）

- **标题：** 都是机器人控制，为什么「走路」用 RL，「干活」却用 Transformer？
- **类型：** blog
- **作者：** 深蓝具身智能（微信公众号；《具身智能基础》专栏第 12 篇；编辑｜阿豹；审编｜具身君）
- **原始链接：** https://mp.weixin.qq.com/s/9prT5Ds0paBthAiupFQTqA
- **发表日期：** 2026-09-12（推断，与入库日同周）
- **入库日期：** 2026-09-12
- **抓取方式：** WebFetch 正文解析（本环境 `wechat-article-for-ai` 未预装）；Jina Reader 对 `mp.weixin.qq.com` 常返回 CAPTCHA，未采用
- **专栏专辑：** [《具身智能基础》](https://mp.weixin.qq.com/mp/appmsgalbum?__biz=MzkwMDcyNDUzMQ==&action=getalbum&album_id=4525948187102363653)（本篇为专辑外延伸第 12 篇）
- **姊妹篇：** [机器人学习五大范式](wechat_shenlan_robot_learning_five_paradigms.md)、[RL 运动控制 pipeline](wechat_shenlan_rl_motion_control_pipeline.md)、[BeyondMimic SciRob 导读](wechat_shenlan_beyondmimic_science_robotics_2026-09-10.md)
- **一句话说明：** 从「训练数据已告诉机器人什么 / 剩余知识如何获得 / 外部注入时哪种表征最划算」三问出发，解释运控（locomotion）为何更依赖仿真 RL 试错，而操作（manipulation）更常从人类示范、Diffusion 与 VLA 预训练出发；并收束 UMI on Legs、BeyondMimic、RLT 三类「手脚分工」合流范式。

## 核心摘录（归纳，非全文）

### 问题重框

- 从电机视角看，走路与操作都是「接收指令 → 驱动关节」，但技术栈关键词分化：**运控**侧 RL / AMP / BeyondMimic；**操作**侧模仿学习 / ACT / Diffusion Policy / VLA。
- 并非「腿用 RL、手用大模型」这么简单；真正决定选型的是 **知识缺口** 与 **获取成本**。

### 三个引导问题

1. **训练数据已经告诉了机器人什么？**
2. **剩下的知识，又可以通过什么方式获得？**
3. **当知识必须从外部注入时，哪种表征方式最划算？**

### 运控 vs 操作：任务结构差异

| 维度 | 运控（如速度跟踪走路） | 操作（如把红杯放进托盘） |
|------|------------------------|--------------------------|
| **目标清晰度** | 方向/速度往往已给定 | 需从场景理解物体、指令与步骤 |
| **数据能直接给的** | 理想姿态/参考动作（像什么样） | 人类示范的完整成功轨迹 |
| **数据给不了的** | 接触动力学中如何稳定实现 | 偏离示范后如何修正 |
| **试错经济学** | 大量失败可在仿真完成 | 物体/接触/奖励建模成本高，从零探索难 |

### 运控：为何人类动作仍要 RL？

- 动作视频是「标准答案录像」，不是可直接运行的控制程序：不告诉每个电机力矩、接触后如何调整、被推后如何恢复。
- 人–机形态差使参考动作只能描述「希望动成什么样」，**怎样做出来** 仍需闭环练习。
- **AMP**：判别器从参考动作学「风格奖励」，与任务奖励一起训练策略——数据回答「什么样算像」，RL 回答「怎样才能做到」。
- **BeyondMimic**：参考动作 → 姿态/速度目标，仿真 RL 练习跟踪；失败率自适应采样、适度域随机等工程细节支撑高动态技能。

### 操作：为何常从人类示范开始？

- 稀疏任务奖励信息不足（抓偏、方向错、未对准等难以从「失败」反推）。
- 逐步写奖励成本高，且易引入捷径行为。
- **ALOHA + ACT**：遥操作示范同时记录图像、状态与动作，提供穿过复杂搜索空间的成功路线。
- **分布偏移**：部署后偏离训练分布，误差累积——示范缩小搜索范围，RL/在线校正处理未覆盖细节。

### 多解性与生成模型

- 合理轨迹多条（绕障左/右），简单平均会得到穿障的错误中间解。
- **Diffusion Policy**：学习动作分布，从噪声逐步生成；执行一段后重新观测闭环。
- **概念澄清**：Diffusion = 生成方式；Transformer = 网络结构；ACT 用 Transformer 但不是语言大模型；VLA 解决更上层的视觉–语言–任务语义。

### VLA 与 π₀

- 固定任务可专学图像→动作；开放场景需互联网规模 VL 先验。
- **OpenVLA**：预训练 VLM + 机器人动作头；窄任务 Diffusion Policy 有优势，多物体/语言条件时 VLA 更擅用语义。
- **π₀**：VLM 表征理解场景 + Flow Matching 动作模块生成连续行为。

### 三类合流（手脚分工）

| 系统 | 上层（任务/语义） | 下层（身体/执行） | 接口 |
|------|-------------------|-------------------|------|
| **UMI on Legs** | 真实示范学末端轨迹 | 仿真 RL 全身协调 | 末端轨迹 |
| **BeyondMimic** | Diffusion 组织 RL 技能分布 | RL 运动跟踪练技能库 | 潜运动表示 + 目标引导 |
| **RLT** | VLA 表征理解任务 | 轻量在线 RL 打磨精细动作 | RL token 从 VLA 特征 |

### 文内收束

- 走路缺的是 **动力学中的身体经验**（目标常已明确）；操作还缺 **物体/指令/步骤理解**（示范与 VL 预训练更划算）。
- 当机器人既要走又要干活，路线正在汇合：各算法做 **训练信号最便宜** 的那一段。

## 文末参考文献（标题级索引）

| # | 文献 | 文内角色 |
|---|------|----------|
| 1 | Peng et al., AMP | 风格判别奖励 + RL |
| 2 | BeyondMimic | RL 跟踪 → 扩散组合技能 |
| 3 | Zhao et al., ALOHA / ACT | 遥操作示范 + 动作块 |
| 4 | Chi et al., Diffusion Policy | 多模态动作分布 |
| 5 | Kim et al., OpenVLA | VLA 开源栈 |
| 6 | Rudin et al., Learning to Walk in Minutes | 并行 RL 运控 |
| 7 | Ha et al., UMI on Legs | 示范上层 + RL 下层 |
| 8 | Ross et al., DAgger | 分布偏移纠偏 |
| 9 | RLT（2026） | VLA + 在线 RL 精细操作 |
| 10 | π₀ | VLM + Flow Matching 动作 |

## 对 wiki 的映射

| 主题 | 关系 |
|------|------|
| [运控 RL vs 操作 IL/VLA 学习栈](../../wiki/comparisons/locomotion-rl-vs-manipulation-learning-stack.md) | **主沉淀页**：知识缺口、试错经济学与三类合流 |
| [RL vs IL](../../wiki/comparisons/rl-vs-il.md) | 双主干对照；本篇补 **任务结构（loco/manip）** 视角 |
| [机器人学习五大范式](../../wiki/comparisons/robot-learning-five-paradigms-taxonomy.md) | 学习信号 taxonomy；本篇补 **身体部位/任务形态** 选型叙事 |
| [BeyondMimic](../../wiki/methods/beyondmimic.md) | AMP/跟踪 + Diffusion 合流代表 |
| [Diffusion Policy](../../wiki/methods/diffusion-policy.md) | 操作多解性与分布建模 |
| [VLA](../../wiki/methods/vla.md) | 语义层与开放任务 |
| [Loco-Manipulation](../../wiki/tasks/loco-manipulation.md) | 全身既要走又要干活的任务域 |
