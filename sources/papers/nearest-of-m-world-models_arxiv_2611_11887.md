# Nearest-of-m 世界模型：一步世界模型需要多少候选（arXiv:2611.11887）

> 来源归档（paper）

- **标题：** How Many Candidates Does a One-Step World Model Need? A Random-Codebook Account of Nearest-of-m Training on Frozen DINO Tokens
- **类型：** paper
- **作者 / 机构：** Yong Hwi Song、Keon Myung Lee（Chungbuk National University，韩国清州）
- **arXiv：** <https://arxiv.org/abs/2611.11887>
- **PDF：** <https://arxiv.org/pdf/2611.11887v1>（v1，cs.LG，2026-11-09）
- **代码：** https://github.com/thddydgnl/nearest-of-m-world-models
- **入库日期：** 2026-11-11
- **一句话说明：** 在冻结 DINOv2/v3 patch token 上训练一步随机世界模型，用「最近样本匹配」（条件 IMLE）取代多步流匹配；把该目标解释为 m 个随机码本的期望失真，回答候选数 m 需要多大、何时任何 m 都不够；规划成功率没有随之提升。
- **核对范围：** 读了 arXiv v1 PDF 全文：摘要、第 1–11 节正文、全部表格（Table 1–18）的文字版、附录 A（推导）、B（I(m,K) 与预先记录的预测）、C（基准构造）、D（实现与消融）、E（读出器）、F（附加结果，含 F.1–F.5）、G（负面结果）、H（统计方法）；图（Figure 1–9）仅读了图注与 pdftotext 抽出的坐标文字，未逐图目视核对；附录证明按文字推演过，未自行重推；参考文献列表只扫了与正文相关的条目；论文 TMLR 投稿稿 `paper/tmlr/` 与仓库 `main/reports/` 未读。仓库只静态读了 README（GitHub API 对本会话未开放，未列出完整文件树，未读代码）。
- **通读：** 已完成

## 开源状态

- README 声明仓库含论文、`main/code/imlewm/` 训练 / 评测 / 记忆化检查 / 规划代码、队列脚本与报告；论文附录 D 写「Code ... will be released with the paper」。仓库实际内容以 README 为准，**未运行也未逐文件核对**。

## 核心摘录

1. **方法（第 3 节）：** 冻结编码器 patch token（N=256，D=384），输入 3 帧上下文 + 动作 + 256 维噪声 z，残差预测下一帧；训练时抽 m 个噪声，无梯度生成 m 个候选，选与真值最近者再带梯度重生成并用 Geman–McClure 损失训练。
2. **随机码本（第 4 节）：** 最近样本损失的期望 = 用 m 个来自生成器的独立样本作码本量化真值的失真 L(q)。
3. **基准（Table 1）：** A1 Bouncing Shapes、A2 弹球（K∈{2,4,8,16} 方向）、Push 分支风险、B 随机 Push-T（隐藏重心）、BAIR；除 BAIR 外均为合成或仿真。
4. **主要结果：** A2-div 上 m=16 覆盖率 0.93 / 精度 0.82，FM-50 为 0.79 / 0.51，延迟 3.3 ms 对 107 ms（Table 2）；Bouncing Shapes F1 误差 7.2 对 12.3 px（Table 3）；随机 Push-T 精度优势显著、召回优势区间含 0（Table 4、15）。
5. **记忆化（第 7 节）：** 非重复上下文中 m=4 出现结果记忆化，excess memorization 与 I(m,K) 拟合 RMSE 0.031，对 (1−1/K)^m 为 0.068；能量分数生成器在任何 m 下都记忆化（Table 2：index 0.87 / 0.97）。
6. **BAIR（第 8 节）：** 样本欠分散，spread-to-error 比 m=4 为 0.22、m=16 为 0.45、m=64 为 0.54；分布指标不及 FM-50（Table 13）。
7. **规划（第 9 节、Table 18）：** 随机 Push-T 成功率 IMLE-WM 31.7% 与回归 31.7%、FM-2 34.7% 相当；分支风险任务 IMLE-WM 风险估计准（0.36 对真值 0.33）但成功率 55.3% 与回归 56.3% 无差，坑率 31.0% 高于回归 24.3%，FM-2 估计最差却成功率最高 67.0%。
8. **局限（第 10 节）：** 7 条，见 wiki 页「局限」。

## 对 wiki 的映射

- [paper-nearest-of-m-world-models](../../wiki/entities/paper-nearest-of-m-world-models.md)
