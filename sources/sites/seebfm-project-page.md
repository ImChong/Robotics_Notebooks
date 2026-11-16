# SeeBFM 官方项目页：逐节阅读与开放状态核对

> 原始项目网页一手资料归档（非论文 PDF 的替代物）

- **标题：** SeeBFM: Behavior Foundation Model That Sees
- **项目页（全文页面）：** <https://capfab.github.io/seebfm-media/>
- **交互式演示：** <https://capfab.github.io/seebfm-media/playground/>
- **作者论文列表：** <https://anindex.github.io/>
- **媒体仓库：** <https://github.com/capfab/seebfm-media>
- **论文发布状态：** 项目页截至 2026-11-11 仍明确标注 `arXiv (coming soon)`；作者主页标记 `Under review · 2026`，尚无可通读的 PDF/附录。**不能将通读本项目网页等同通读论文**。
- **代码：** 公开网站 HTML、幻灯片、视频和 Three.js 预录仿真回放前端。未检索到 SeeBFM 策略训练/推理、学生回归模型、checkpoint 或真机部署源码。
- **数据：** `playground/data/` 包含网站复现预录仿真轨迹的索引/动作/机器人/地形；未证实为算法重训数据集。
- **核查日期：** 2026-11-11
- **核对范围：** 已完整阅读项目主页公开的从概述、How it works、结果、P-BFM 对照、What we learned、Cameras 到 Limitations/Credits 的正文及表格（网页 174 行文本）；同步读完 `slides/spec.js` 的 15 段讲解/数值、`media/overview.vtt` 的完整字幕，以及 `playground/index.html`、`playground/viewer.js`、`playground/match.js` 等官方回放源文件。**未阅读论文 PDF：作者尚未公开**。
- **通读：** 已完成

## 一、项目动机与结构

- BFM-Zero 可以通过 reward、goal pose、motion 等生成 latent prompt，却主要从平地运动与自身状态学习；遇到楼梯和箱体时同一 prompt 不足以决定步态。
- 后训练把高度扫描接入执行和评判相关网络，**不接入将奖励/目标映射到 prompt 的 backward 网络**；官方主页以 “Prompt the what, perceive the how” 概括这条边界。
- 官方主页的量化设计描述：**379 h** 总运动数据（与 BFM-Zero **2.4 h** 对比），其中 **78 h** 是地形运动；机器人附近高度扫描为 **598 点**；页面报告 **97.7%** 的下一步落脚位置处于扫描区域内。以上为项目网页宣称数值，尚无公开论文供核对采集/统计细节。
- 双深度相机学生通过模拟 rollout 渲染的相机观测估计高度扫描，替代仿真直接提供的 scan；控制策略冻结，不需要重新训练。

## 二、主要仿真评测：指标和口径分开

### A. 官方视频中六个 locomotion prompts 汇总

| 地形 | SeeBFM crossing | BFM-Zero crossing |
|---|---:|---:|
| 上楼梯 | 54.4% | 0% |
| 箱体 | 41.0% | 0% |
| 下楼梯 | 68% | 5% |

上楼梯扫描消融：真实扫描 54.4%、level scan 4.0%、no-scan 训练 0%。此组出自讲解 `slides/spec.js`、`overview.vtt`，**不是**下表指定 forward prompt 的跨越率。

### B. 项目主页的单独任务与 P-BFM 对照

下表来自官网 “Compared with P-BFM” 正文；步高 0.10–0.15 m、训练扰动关闭；`walking` 汇总了四个 walking prompts，`forward` 仅为 0.7 m/s 前行 prompt：

| 模型 | 感知输入 | forward：上/箱/下 crossing (%) | walking：上/箱/下 crossing (%) | walking：上/箱/下 fell (%) |
|---|---|---|---|---|
| SeeBFM | 高度扫描 | 100.0 / 81.2 / 97.0 | 33.6 / 35.6 / 38.5 | 0.9 / 13.5 / 1.4 |
| SeeBFM 深度学生 | 双深度相机 | 98.3 / 85.4 / 97.2 | 43.7 / 35.0 / 43.0 | 24.3 / 35.0 / 16.3 |
| P-BFM，仅 prompt | 深度相机 | 3.5 / 92.4 / 99.3 | 2.5 / 42.7 / 62.7 | 10.7 / 15.9 / 6.0 |
| P-BFM，另加指令 | 深度相机 | 97.2 / 84.7 / 93.1 | 38.6 / 39.5 / 71.4 | 27.5 / 41.7 / 15.3 |

**局限：** 官方明确说明两模型分别在各自模拟器执行；不能将此表视为完全统一环境中的控制变量比较。穿越成功、停止前进和摔倒应分开计量。深度学生的跌倒率高于使用真 scan 的 SeeBFM。

### C. 其他实验观察

- 官方报告上楼梯 **motion tracking**：让 backward prompt 网络接收扫描会使误差 **1.14 → 1.31**（官网没有在该句给出误差单位）。
- 去掉 prompt 网络的 average 后，避免各 prompt 倾向相同默认动作；不做该处理时侧走几乎停滞。
- 地形速度保持比例：前行 76%、转向 72%、侧行 42%、后退 12%（官方视频）。
- 官网语言测试在平地上：“已选中的动作”完成率 **99.6%**，失败主要来自上层语言选择；语言系统使用 Qwen3-8B 选运动、all-MiniLM-L6-v2 做句向量匹配。公开网页的输入仅能匹配录制过的句子，而非在线策略生成任意动作。
- 官方网页声明记录了 **19,232** 个仿真尝试，并通过 Playground 回放；不能称为 19,232 次真机实验。
- 官网局限：楼梯/箱体上的侧行和倒退会变慢或停住，深度学生下楼梯最弱，已录制楼梯攀爬动作仅 **68/257** 跟踪；真机 G1 试验尚在准备。

### D. 高障碍表述存在作者原始材料差异

- **主页 Limitations：** “Blocks taller than about 25 cm are not climbed.”
- **演示讲解 `slides/spec.js`：** 描述 45 cm 方块前停下，并称高于 35 cm 时不尝试。
- 两处数值阈值并不一致，且 “不爬越” 与 “不尝试” 也并非同一指标，**先并列保留**，不自行断定唯一能力极限。

## 三、开放程度与复现结论

- **网站和回放代码已公开：** `capfab/seebfm-media` 中有 `slides/`、`media/`、`playground/`、`playground/data/`。
- **算法代码未确认开放：** 项目页无 SeeBFM 控制模型训练/推理入口，学生网络源码与参数未发布。
- **真机状态：** 所有上述结果来自模拟；项目页明确“Everything so far is in simulation; tests on a real Unitree G1 are in preparation.”

## 交叉引用

- [论文 PDF/附录缺失及待办](../papers/seebfm_preprint_2026.md)
- [官方回放源码核对](../repos/seebfm-media-page-source.md)
- [SeeBFM 知识节点](../../wiki/entities/paper-seebfm.md)
