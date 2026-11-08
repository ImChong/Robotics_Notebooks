# SimpleVLA-RL: Scaling VLA Training via Reinforcement Learning（arXiv:2509.09674）

> 来源归档（paper）

- **标题：** SimpleVLA-RL: Scaling VLA Training via Reinforcement Learning
- **类型：** paper（技术报告，"SimpleVLA-RL Technical Report"，版首日期 2025-09-12）
- **arXiv：** <https://arxiv.org/abs/2509.09674>（v1，2025-09-11 提交，cs.RO）
- **PDF：** <https://arxiv.org/pdf/2509.09674>
- **代码：** <https://github.com/PRIME-RL/SimpleVLA-RL>（论文脚注 3：代码 2025 年 5 月公开）
- **作者单位：** 清华大学、上海 AI Lab、上海交通大学、北京大学、香港大学
- **入库日期：** 2026-11-08
- **一句话说明：** 基于 veRL 的 VLA 在线强化学习框架：二值结果奖励 + GRPO 类组相对优势（去 KL、clip-higher、动态采样、温度 1.6），作用于 OpenVLA-OFT，在 LIBERO / RoboTwin 1.0 / 2.0 与 sim2real 上超过 SFT。
- **核对范围：** 下载 arXiv v1 PDF 全部 24 页，pdftotext 后逐段通读正文第 1–8 节（引言、预备知识、方法、实验设置与主结果、数据稀缺 / 泛化 / 真机分析、pushcut 与失败模式讨论、相关工作、结论）及全部表 1–7、图 1–5 的文字与图注；参考文献列表仅浏览未逐条核对；论文无附录；未运行任何代码，未读 GitHub 仓库实现，图中像素内容（如图 5 的截图）只依据图注与正文描述。
- **通读：** 已完成

## 开源状态

- **已开源**：论文脚注 3 给出 `PRIME-RL/SimpleVLA-RL`，并称 2025 年 5 月公开代码（本次未打开仓库核查，权重 / 数据开放情况未核）。

## 核心摘录（均来自论文，节号为论文节号）

1. **问题（§1）：** VLA 的 SFT 扩展受限于（1）人类遥操作机器人轨迹稀缺昂贵，（2）分布偏移下泛化差。动机是把 DeepSeek-R1 式的结果奖励在线 RL 搬到 VLA。
2. **框架（§1、§3）：** 构建在 veRL 上，加入 VLA 专用的交互式轨迹采样、并行多环境渲染、训练 / 推理 / 渲染一体化与损失计算。流程见图 2：对每个输入多次随机采样轨迹 → 环境给 0/1 结果奖励 → 用奖励与动作 token 概率算 GRPO 损失更新策略。
3. **动作表示（§3.1）：** 采用输出动作 token 分布的方式（文中称与 PPO 类 RL 算法最兼容），随机采样产生多样轨迹；扩散式与 MLP 回归式解码未采用。
4. **奖励（§3.2、式 9）：** 轨迹成功为 1，否则为 0，并均匀传播给该轨迹所有动作 token；不使用过程奖励。
5. **探索增强（§3.3）：** 动态采样（丢弃全成功或全失败的组，式 10）、clip 上界由 1.2 放宽到 1.28、rollout 温度由 1.0 提到 1.6；引言称这些带来 10–15% 的一致提升（图 3 为 LIBERO-Long 上的消融曲线）。
6. **训练目标（§3.4、式 11–12）：** 采用 GRPO（Shao et al., 2024），优势为组内 (R_i − mean)/std，PPO 式裁剪 ratio，并按 DAPO 去掉 KL 项（省去参考模型）；无价值网络。
7. **设置（§4.1）：** OpenVLA-OFT 为骨干，但与官方实现不同：仅单视角图像 + 语言 + 本体状态（LIBERO 不用本体状态），只保留并行解码与动作分块，用 LLaMA2 输出头 + 交叉熵生成动作 token，官方用 MLP 输出连续动作 + L1 回归，因此无法使用官方 checkpoint，需从头 SFT。8×A800 80GB，lr 5e-6，训练 batch 64，采样数 8，mini-batch 128，ε_low 0.2，ε_high 0.28，T=1.6；动作块数 LIBERO 8、RoboTwin 25；评测用贪心采样，每个基准测 3 次。
8. **主结果（§4.2，表 2–4）：** LIBERO 四套件平均 91.0 → 99.1（LIBERO-Long 86.5 → 98.5）；RoboTwin 1.0 四任务平均 39.8 → 70.4；RoboTwin 2.0 十二任务平均 38.3 → 68.8（文中对比 π0 49.2、RDT 33.3）。
9. **数据稀缺（§5.1，表 5）：** 每任务仅 1 条示范的 SFT 后（LIBERO 平均 48.9，Long 17.3），RL 后平均 96.9（Long 91.7）；全量示范 SFT + RL 为 99.1。
10. **泛化（§5.2，图 4）：** 每套件 9 个已见任务训练、1 个未见任务评测；SFT 在多数未见任务上退化、常降到 0%，RL 在未见任务上多数提升（正文例：LIBERO-Spatial Unseen Task 1 由 43.3% 到 71.8%）。
11. **真机 sim2real（§5.3，表 6）：** 四任务（Stack Bowls、Place Empty Cup、Pick Bottle、Click Bell；表 6 列名），双 AgileX Piper，纯仿真数据（1000 条仿真轨迹 SFT + 1000 个仿真场景 RL），每任务 50 次试验，平均 17.5% → 38.5%，RDT 23.5%。
12. **pushcut（§6.1，图 5）：** RoboTwin 2.0 的 move can pot、place a2b left/right 中，示范均为抓取-移动-放置，RL 后策略学到直接推送；作者类比 DeepSeek-R1 的 "Aha Moment"。
13. **失败模式（§6.2，表 7）：** 基础模型 0 条轨迹 SFT 时成功率 0，RL 完全无效（无成功样本、只有结果奖励）；100 条 SFT 平均 7.3 → 25.4，1000 条 SFT 28.2 → 50.4；初始成功率过低时增益很小（pick dual bottles：100 条 1.2 → 4.3）。
14. **与同期工作（§7.2）：** 作者称本工作是最早的系统性 VLA 在线 RL 探索之一；列举 RIPT-VLA（RLOO）、RLinf、VLA-RL（PPO）、TGRPO、RFTF 等。

## 文内数字不一致处（读到的，未下结论）

- 引言贡献（§1）称 LIBERO-Long 单示范 "从 17.1% 到 91.7%"，§5.1 与表 5 为 17.3% → 91.7%。
- §5.3 正文把真机任务写作含 Handover Block，表 6 列名为 Place Empty Cup；本归档按表 6 记录。
- Stack Bowls：正文称 SFT 32% → 70%，表 6 的 SFT 为 38.0，二者不一致。

## 对 wiki 的映射

- [paper-simplevla-rl](../../wiki/entities/paper-simplevla-rl.md)
