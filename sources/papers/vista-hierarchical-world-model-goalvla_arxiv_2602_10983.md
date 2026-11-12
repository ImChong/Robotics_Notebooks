# Scaling World Model for Hierarchical Manipulation Policies（VISTA，arXiv:2602.10983）

> 来源归档（paper）

- **标题：** Scaling World Model for Hierarchical Manipulation Policies
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2602.10983>（v1，2026-02-11，cs.RO）
- **PDF：** <https://arxiv.org/pdf/2602.10983v1>
- **项目页：** <https://vista-wm.github.io>（摘要页给出；本次未访问，未见代码仓库地址）
- **作者与机构（首页）：** Long Qian, Yueze Wang, Jiaxi Song 等；西安交通大学、北京智源人工智能研究院（BAAI）、清华大学、新加坡国立大学、中国科学院自动化研究所
- **入库日期：** 2026-11-12
- **一句话说明：** 用大规模预训练的世界模型（Emu3.5 继续训练）做高层规划，生成「文本子任务 + 多视角目标图像」序列，再由目标图像条件化的 VLA（GoalVLA，π0 架构）执行；真机 OOD 物体泛化上 π0 的 14% 提到 69%。
- **来源线索：** Glen Berseth《My CoRL 2026 Reading List》第 076/101 条（P2，World Models & World Action Models 组）；清单一句话摘要为 LLM 据摘要起草，排序为作者个人兴趣模型打分，非论文质量评价。

## 核对范围与通读

- **核对范围：** 下载 arXiv v1 PDF 并用 pdftotext 逐段读完正文 I 至 VII 节（含全部公式 1 至 4、Table I），读附录 A（实现细节）、B（数据集构造与 Table A.1）、C 至 G 的文字说明；Fig. 5 至 8 的生成样例与 Fig. 10 只看了图（Fig. 10 平均值为从柱状图目测，非文中数字）；未读参考文献全文，未访问项目页与视频，未见开源代码，未复现任何实验。
- **通读：** 已完成

## 核心摘录（作者自述，带章节）

1. **问题与动机（I）：** VLA 在 OOD 与真机数据稀缺时脆弱；文本子目标缺空间物理约束，稠密视频预测有时间漂移与物理不一致；提出只预测稀疏关键帧作为「视觉里程碑」。
2. **形式化（III，式 1）：** (l_i, g_i) = W(L, h)，a = π_θ(I_t, l_i, g_i)；机器人执行直到当前观测与 g_i 在视觉上对齐，再更新历史并向 W 请求下一阶段。
3. **世界模型（IV-A，附录 A-A）：** 在开源 Emu3.5 检查点上继续训练 2000 步；图像用 IBQ tokenizer（512×512 输入，1024 视觉 token），文本用 Qwen3 tokenizer，统一词表 282,926；总参数 34.1B（transformer 31.2B + embedding 2.9B）；自回归交错生成文本子任务与目标图，推理用 beam search；多视角图像按固定顺序展平；128 块 H100 训练约 2 天，全局 batch 512，学习率 1e-5。
4. **GoalVLA（IV-B，附录 A-B）：** 按 π0 架构，PaliGemma-3B 骨干 + 0.3B 动作专家，输入 3 张当前图加 3 张目标图（SigLIP 编码）、子任务文本与末端 6D 位姿；流匹配（式 4），10 步解码出 30 步动作块，真机只执行前 10 步，取第 5 与第 10 步作路点；两项训练技巧：子任务感知动作补零（子任务边界后补零使策略学会停止）与随机目标图偏移（边界重叠窗口内随机用 g_i 或 g_{i+1}）。
5. **数据（IV-C，附录 B，Table A.1）：** 把 1.2M 轨迹自动拆成交错的子任务与目标图序列（50 个原子技能库；RDP 算法与夹爪状态切分；Qwen2.5-VL 72B 合并并写子任务描述），正文称 15.2B token、14 个 embodiment，附录 B 写 10 个 embodiment；来源 Open-X Embodiment、AgiBot World Beta、自采 Mobile Aloha；另共训 15.0B token（正文）的 Any-to-Image 数据（Table A.1 写 15.2B，数字对不上）。
6. **GoalVLA 训练（附录 A-B）：** 第一阶段用 AgiBot-Beta 的 200,000 条带子任务与目标图标注的样本训 100,000 步；第二阶段在 737 条自采真机轨迹上微调 10 epoch（约 20k 步）。
7. **真机评测设置（V-A，附录 D）：** 训练仅 2 小时、5 件物体（鸡蛋、可乐罐、苹果、牛奶盒、可颂）的抓放，每任务约 150 条；基线为 π0 与把指令换成子任务的 π0-subtask；域内 15 个场景（5 任务 × 3 设置）、OOD 63 个场景（21 个未见物体 × 3 种桌布与布局），每场景 3 次，共 78 × 3 = 234 次 rollout / 策略。
8. **主要数字（V-C，Table I）：** 基础设置执行成功率平均 π0 0.96、π0-subtask 0.91、本文 0.93；未见干扰物 0.73 / 0.78 / 0.82；未见目标物 approach 0.40 / 0.73 / 1.00，execution 0.04 / 0.31 / 0.67。摘要与引言的 14% 到 69% 对应 63 个新场景（Fig. 10 的 Avg 柱，图上目测约 0.14 与 0.70；文字里只写 14% 与 69%）。
9. **局限（VII）：** 真机评测只有抓放；生成的子目标可能与真实子任务终点有小偏差，偏差大时 GoalVLA 失败；执行仍依赖真机训练数据里有对应交互模式。附录 F 自述：世界模型的语义 / 空间指令跟随是继承自 Emu3.5 的部分能力，仍有限且常出现幻觉。

## 与清单摘要的核对

清单摘要："Hierarchical VLA: a pretrained world model generates subgoal images and a VLA executes them, improving OOD generalization."——结构与方向与论文一致；细节缺口见 [paper-vista-hierarchical-world-model-goalvla](../../wiki/entities/paper-vista-hierarchical-world-model-goalvla.md)「与清单说法的核对」一节。

## 对 wiki 的映射

- [paper-vista-hierarchical-world-model-goalvla](../../wiki/entities/paper-vista-hierarchical-world-model-goalvla.md)
