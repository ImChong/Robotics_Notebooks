# perceptron_isaac_05

> 来源归档（blog）

- **标题：** Introducing Isaac 0.5
- **类型：** blog
- **来源：** Perceptron Inc. 官方博客
- **原始链接：** https://www.perceptron.inc/blog/introducing-isaac-0-5
- **技术报告 PDF：** https://pub-d90b81cad7254a1aa6b148ac18153c0c.r2.dev/isaac-0.5.pdf
- **代码：** https://github.com/perceptron-ai-inc/isaac
- **权重页：** https://huggingface.co/PerceptronAI/Isaac-0.5
- **入库日期：** 2026-08-27
- **最后更新：** 2026-08-27
- **一句话说明：** Perceptron 发布 **Isaac 0.5**：36B 稀疏具身基础模型（期望 **2.5B** 激活），把视频理解、空间 grounding、任务进度与机器人控制收进同一共享骨干；给出 **无动作标签视频 ↔ 遥操作** 的 scaling law（1,000 h 视频需约 5,884 h teleop，1M h 视频只需 28 h，约 **210×**）。

## 开源状态（项目页 / Hub / GitHub 核查，2026-08-27）

| 项 | 状态 |
|----|------|
| 代码 | **已开源** Apache 2.0 — [perceptron-ai-inc/isaac](https://github.com/perceptron-ai-inc/isaac)（LeRobot 子模块 `perceptron_isaac`） |
| 权重 | Hub 页截至入库日标 **COMING SOON**（[PerceptronAI/Isaac-0.5](https://huggingface.co/PerceptronAI/Isaac-0.5)）；技术报告宣称将释放权重 |
| mHarmony / TensorStream | **未纳入** `perceptron_isaac` extra；干净 checkout **不足以** 渲染 / 训练 / 推理 |
| 未来 percept 目标构造与损失 | **专有**（论文公开目标形式，不公开 generator / ℓ 实现） |
| 训练数据 | **未发布** |
| 结论 | **部分开源**：训练/推理代码与 LeRobot 入口已公开；权重与数据栈运行时仍有缺口 |

交叉归档：[perceptron_isaac_05.md](../papers/perceptron_isaac_05.md)（技术报告）、[perceptron-isaac.md](../repos/perceptron-isaac.md)（仓库）、[perceptron-inc.md](../sites/perceptron-inc.md)（公司站）。

## 核心摘录

### 1) 产品定位

- **36B** 稀疏模型（Qwen-family 视觉–语言骨干 + null-expert 路由）；期望 **2.5B** 激活参数 / token。
- 输入：图像、视频、语言指令、机器人状态、既往动作。
- 输出：视频问答、指向 / 跟踪、任务进度，以及 **FAST 离散动作** 与 **Flow/DiT 连续动作块**。
- 训练数据叙事：**35+** 本体、**100k h** 机器人经验、**1M h** 通用视频、**3T** 多模态 token；视频理解、空间 grounding、任务进度、未来 percept 与动作 **从一开始共训**。

### 2) Video–teleop scaling law（博客 + 技术报告对齐）

固定 **80:30:30** 通用视频 : egocentric : UMI，缩放预训练并在 held-out 轨迹上测动作预测损失。

- 目标损失 **2.50**：1,000 h 通用视频 → 约 **5,884 h** teleop；1M h 通用视频 → **28 h** teleop（**210.3×**；相邻 rung 括号 **83–300×**）。
- 交互：仅 1 h teleop 时，10× 视频约降损失 **0.006**；从约 **100 h** teleop 起，同样 10× 视频约降 **0.21**。
- 读法：便宜视频可置换昂贵遥操作，但 **需要足够动作 grounding** 后视频才更有效。

### 3) Semantic world modeling（未来 percept）

- **Percept** = 任务相关的可见状态/变化（物体状态、空间关系、affordance、任务阶段、接触变化等）。
- 监督来自未来观测的自动构造，**无需人工标注或动作标签**；目标构造与损失实现 **专有**。
- 与像素重建 / JEPA 特征预测 / 直接动作预测并列：percept 坐在 **像素与动作之间**，更新同一共享骨干。

### 4) 接口与系统

- **mHarmony**：基于 OpenAI Harmony 的 typed 多模态事件编译器；再打成 **TensorStream** packed tensor。
- **Null experts**：token 可选 0–8 个真实专家（256 专家库）；动态算力。
- **RTC 训练**：闭环时在当前 chunk 执行中预测下一 chunk（既往动作 + 最新观测）。
- 对照表（博客自报）：相对 π0.7 / π0.5 / Qwen-VLA / LingBot-VLA / MolmoAct2 / SmolVLA / Octo / OpenVLA，Isaac 同时勾选 **35+ 本体、RTC、prev. actions、mistake modeling、非机器人视频、Flow expert、开源**。

### 5) 评测数字（博客摘要；细节以技术报告为准）

- Grounding：ScreenSpot-Pro **62.6**、LVIS Count **32.8**、CARPK **19.1**（对照最强同 harness Qwen3-VL 跑分 54.8 / 28.7 / 6.0）。
- 推理成本叙事：三图请求 **26.9 TFLOP** vs 最强对照 **228.4**（约 **8.5×** 更低）。
- 国际象棋操作 one-epoch / 单 expert episode：Isaac 损失降幅 **10.5× / 9.5× / 7.0×**（固定着法 / 记号条件 / 防守变例），π0.5 最近为 **3.1× / 2.6× / 2.3×**。

## 对 wiki 的映射

- [perceptron-isaac-05](../../wiki/entities/perceptron-isaac-05.md) — 实体页（新建）
- [vla](../../wiki/methods/vla.md) — 开源通才 VLA 对照
- [foundation-policy](../../wiki/concepts/foundation-policy.md) — 具身基础策略谱系
- [embodied-scaling-laws](../../wiki/concepts/embodied-scaling-laws.md) — 视频小时置换 teleop 的 scaling 案例
- [lerobot](../../wiki/entities/lerobot.md) — `policy.type=perceptron_isaac`
- [perceptron-egocentric](../../wiki/entities/perceptron-egocentric.md) — 同机构 Mk1 标注 API
- [isaac-gr00t](../../wiki/entities/isaac-gr00t.md) — **消歧**：NVIDIA Isaac ≠ Perceptron Isaac

## 当前提炼状态

- [x] 博客核心摘要与开源核查
- [x] 技术报告 PDF 数字对齐（LIBERO 适配、激活参数、数据混合物）
- [x] wiki 实体页映射确认
- [ ] Hub 权重实际可下载后再把开源结论从「部分」升为「已开源」
