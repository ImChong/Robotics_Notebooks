# MAESTRO（arXiv:2511.00917）

> 来源归档（paper）

- **标题：** MAESTRO: Orchestrating Robotics Modules with Vision-Language Models for Zero-Shot Generalist Robots（arXiv v1 首页标题；CoRL 2026 清单所写标题为 "…with Coding Agents…"，疑为后续版本改名，推测）
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2511.00917>
- **PDF：** <https://arxiv.org/pdf/2511.00917v1>
- **项目页：** <https://maestro-robot.github.io>（论文首页给出；本次未读，也未见 GitHub 仓库链接）
- **作者 / 机构：** Junyao Shi*、Rujia Yang*、Kaitian Chao*（共同一作）等，宾夕法尼亚大学（含 Dinesh Jayaraman、Kostas Daniilidis、Pratik Chaudhari）
- **入库日期：** 2026-11-12
- **一句话说明：** 以 VLM 编程智能体为核心的模块化零样本通用机器人策略：在闭环「规划—反应—重规划」中写代码调用感知、几何、控制、学习策略与图像编辑工具；真机桌面任务上多数优于 π0 / π0.5（DROID 微调版）。
- **核对范围：** 读了 arXiv v1 PDF 全文（正文 I–V 节、表 I–IV、图 1–5 的图注与可见文字、附录 A 模块细节、附录 B 评分细则、附录 C 相关工作扩展讨论、参考文献列表）；未读项目页与补充视频，论文未给出代码仓库；PDF 中图内像素内容仅看了图注与可抽取文字，未逐图细看。
- **通读：** 已完成

## 核心摘录（均出自论文，章节为 v1 编号）

1. **方法（III）：** MAESTRO 是写并执行代码的智能体，工具覆盖感知、几何、控制、预训练策略、图像编辑；每个子步后 VLM 判断子目标是否达成，成功则规划下一步，失败则诊断并重写同一子步代码（III-B）。
2. **VLM 选择（IV-A）：** 实验用 Gemini Robotics-ER 1.5 作 MAESTRO 的 VLM；附录 A 另述关键点选择工具用 GPT-o3，VLA 监控用本地 Qwen2.5-VL-72B-Instruct（2 Hz 是/否判断）。
3. **平台（IV-A）：** 桌面为 DROID 配置（Franka Panda 7-DoF + Robotiq 2F + 腕部与第三人称相机）；移动为 Unitree Go2-W 轮足 + AgileX PiPER 机械臂。7 个桌面任务、4 个移动任务。
4. **评测协议（IV-A、附录 B）：** 按 STAR-Gen 分类法生成扰动，每任务共 5 次试验（1 初始 + 4 生成），所有方法相同；指标为按子步骤打分的进度（0–100），沿用 π0 的做法。
5. **桌面结果（表 II，进度均值 ± 标准差）：** MAESTRO 对应 Gemini Robotics Agent / π0 / π0.5：放入碗 98.0 vs 73.3 / 74.0 / 70.0；叠毛巾角 71.3 vs 40.0 / 47.0 / 70.0；开柜 68.0 vs 3.3 / 8.3 / 0.0；转方块 60.0 vs 23.6 / 29.0 / 10.0；切香蕉 92.0 vs 71.0 / 30.0 / 14.0；挂杯 69.0 vs 46.0 / 59.0 / 80.0；擦白板后叠杯 63.0 vs 26.7 / 12.0 / 22.0。7 个任务中 6 个 MAESTRO 最高，挂杯任务 π0.5 最高。
6. **移动结果（表 III，仅 MAESTRO，无基线）：** 收集玩具 85.0，扔球入垃圾桶 76.7，搜寻物体并放桌 96.0，按按钮开门 93.3。
7. **消融（表 IV，仅叠毛巾与转方块）：** 去掉高级感知（关键点 + 主动感知）40.0 / 25.0；去掉几何模块 67.5 / 42.5；完整 71.3 / 60.0。
8. **演化（IV-E、III-C）：** 以历史代码、标准输出与 Gemini 对视频的成败分析作上下文示例；开柜最差一次 35% 起，一次演化后 70.0 ± 5.0，第三次后 85.0 ± 7.4。
9. **作者自述局限（IV-B、V）：** 挂杯任务复杂的对象可供性与空间朝向思维链仍会失败；更精细连续控制需要更丰富的低层行为；VLM API 响应延迟带来停顿，运行时间长。
10. **未在 v1 表中出现：** 正文称比较了「MAESTRO + π0.5」，但表 II 没有这一列，v1 中无该组合的数字。

## 对 wiki 的映射

- [paper-maestro](../../wiki/entities/paper-maestro.md)
