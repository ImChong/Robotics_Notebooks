# TRABot（Talk, Render, Act）论文来源归档

## 元数据

- **题目：** Talk, Render, Act: Integrating Social Gesture and Digital Face with Synchronized Speech for Conversational Humanoid Robot
- **方法：** TRABot
- **作者：** Jin Jiang, Kun Li, Jiancong Ma, Shengcai Liao
- **单位：** 阿联酋大学计算与人工智能学院（College of Computing and Artificial Intelligence, United Arab Emirates University）；论文 HTML v2 作者栏与官方项目页均明确列出
- **预印本：** https://arxiv.org/abs/2610.06153
- **HTML：** https://arxiv.org/html/2610.06153v2
- **版本：** v1 2026-10-05，v2 2026-10-09，cs.RO
- **项目页：** https://trabot-humanoid.github.io （归档见 [trabot-project.md](../sites/trabot-project.md)）
- **代码：** 截至 2026-11-13 再核查仍未发布（项目页标注 “Code (soon)”）
- **入库日期：** 2026-10-12；全文复核：2026-11-13
- **核对范围：** arXiv HTML v2 全文——Abstract、§I Introduction、§II Related Work、§III-A–D（含式 1–14、Algorithm 1、Table I）、§IV-A–C（Tables II–VI、12 对话、20 人用户研究、200 回复自动评测）、§V Conclusion、Acknowledgments、References；并对照官方项目页 Method/Results/Code 标签。
- **通读：** 已完成（2026-11-13，针对 v2 全文）

## 论文要解决的问题

对话机器人通常把语音、面部动画和肢体手势分别生成，语义与时间容易错位。TRABot 把手势做成机器人可执行、经过筛选的「运动原子」，由规划器按回复的语义顺序和估计时长选择，再与流式语音、数字人脸统一同步，在真实 Unitree G1 上运行。

## 方法与训练信息

四个 agent：
1. **Motion Atom Construction Agent（离线）：** 从 BEAT2 长时动作数据（重定向到 29-DoF G1，根固定，控制 14 个手臂关节）切出运动原子并审核。
2. **Dialogue Agent（在线）：** 用 Gemini 3.1 Flash Live 产生流式回复音频、有序的交际功能列表和估计回复时长（论文未单独说明 TTS，语音大概率来自该模型）。
3. **Motion Planning Agent（在线）：** 语义条件组合规划器 + 状态感知序列执行器。
4. **Facial Animation Agent（在线）：** 用 SoulX-FlashHead 从同一回复音频生成面部帧，25 FPS；论文未说明 G1 上显示面部的硬件。

运动原子构建：
- 25 Hz 信号上做边界检测，可执行轨迹保留原始 30 Hz；活动度按关节范围归一化，每段录音使用百分位阈值；起始需持续 0.12 s 超阈、结束需 0.16 s 低于阈；小于 0.12 s 的间隙合并，1.0 s 内的收回动作并入手势；边界精修窗 0.32 s、代价上限 0.50、片段 0.4–4 s。
- 可行性限制：关节限位留 0.05 rad 余量；速度/加速度/加加速度上限 7 rad/s、120 rad/s²、2200 rad/s³；接触深于中立位 2 mm 以上则拒绝；用固定根 MuJoCo 回放检查。
- 修复：幅度缩放、平滑、时间拉伸；需拉伸超过 3 倍或原子长于 10 s 的配方丢弃。
- 语义：Qwen3-VL-8B-Instruct 提出标注，再经人工审核；词表 17 种交际功能、8 个类别（强调、枚举、对比、指代、描绘、话轮管理、社交、立场）。原子库规模原文未报告，人工审核拒绝率也未报告。

规划与执行：
- 动态规划求在预算内可行的最大功能前缀 L*；beam search（宽度 32）选原子，排序偏好剩余预算更小、重复更少、左右手平衡。
- 预算 B = 3·T̂，保护间隔 0.35 s；代价还包括起始桥接、原子间过渡、最终回中立位；桥接采用最小加加速度五次曲线，**速度上限为 1 rad/s**（论文 §III-C.3 式 13–14 / §IV-A；旧归档误写 11 rad/s）。
- 只执行第一个原子，再按实测状态重规划剩余功能；另有 force-first 模式。

## 论文报告的结果

- **实时同步（12 条回复）：** 面部偏移 37.5 ms，身体偏移 −666.6 ms（身体先于语音启动），语音–身体重叠 84.72%，12 条全部无失败完成。
- **节拍对齐（BAS，12 条回复，均值±标准差）：** Simple Motion 0.297±0.206 / 0.210±0.129（二值 / 高斯 σ=0.30 s）；Random Motion 0.281±0.197 / 0.190±0.145；Full 0.308±0.141 / 0.225±0.130。提升很小（约 0.011–0.035）。
- **用户研究（20 人，每人 12 轮，1–5 平均排名）：** 自然度 / 表现力 / 一致性——Speech 1.10 / 1.10 / 1.10；Speech+Face 2.10 / 2.10 / 2.08；Simple Motion 3.15 / 3.13 / 3.16；Random Motion 3.87 / 3.91 / 3.87；Full 4.79 / 4.76 / 4.79。手势–语音一致性：Full 4.77，Random 3.86，Simple 3.15。偏好率：Full 84.58%（240 个人次决策中 203 次），Random 13.75%。
- **规划器自动评测（200 条固定回复）：** 平均规划原子数 1.705（Full）vs 1.451（Random Match），+17.5%；未用预算 0.993 s vs 2.376 s，−58.2%。

## 复现与解释边界

- 无代码；原子库规模、人工审核拒绝率、显示硬件均未报告。
- 可行性检查基于固定根 MuJoCo，不含真机站立/行走扰动；身体行为多样性受原子库覆盖限制。
- 用户研究 20 人，比较条件含 Random Motion 但无规划器分项消融；节拍对齐差距小。
- 机构已由论文 HTML v2 作者栏与官方项目页核实：College of Computing and Artificial Intelligence, UAEU。

## 公式与案例复核（arXiv v2）

- 式 (1) 语义动作请求：`R_m = (f_1:K, T_hat)`；式 (2) 原子由**人工确认的功能标签、关节轨迹、持续时间、左右手属性**组成。
- 式 (11) 完整规划代价：`J = C_tr(current, first) + Σ atom_duration + Σ inter_atom_transition + C_return_to_neutral + G`；式 (12) 要求 `J ≤ B(T_hat)` 并最大化有序前缀长度。
- 式 (13–14) `s(u)=10u³−15u⁴+6u⁵`，桥接时间满足 `T_b ≥ 1.875 max_j |q1,j-q0,j|`（速度上限 1 rad/s）。
- Table II 例子：“What are three ways to relax?” → `enumerate → enumerate → enumerate → neutral`；“Coffee or tea—which is better?” → `contrast_compare → uncertainty → neutral`。
- Table V 是**每个回复内五条件的 1–5 排名**，不是独立问卷量表；Full 获 203/240 首选。Table VI 的 Random Match **每个固定回复重复 50 次试验**。
- §IV-A 说明对话使用 Gemini 3.1 Flash Live 产生流式音频，未独立描述另一个 TTS 系统；**不能猜测额外语音引擎**。
- 研究约束：固定根、仅 14 个手臂关节；单组 20 人 × 12 回合用户研究；声学节拍 BAS 差距有限，不能推断广义真机任务成功率。

## Robotics_Notebooks 映射

- 详情节点：wiki/entities/paper-trabot-talk-render-act.md
- 概念入口：wiki/methods/humanoid-voice-interaction.md

## 一手来源

- arXiv 摘要与版本：<https://arxiv.org/abs/2610.06153>
- arXiv HTML 全文：<https://arxiv.org/html/2610.06153v2>
- 项目页：<https://trabot-humanoid.github.io>
