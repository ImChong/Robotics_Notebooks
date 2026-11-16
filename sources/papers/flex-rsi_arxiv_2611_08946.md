# Flex-RSI（arXiv:2611.08946）

> 来源归档（paper）

- **标题：** Flex-RSI: Benchmarking Recursive Self-Improvement of Frontier Agents
- **类型：** paper（技术报告，cs.AI）
- **arXiv：** <https://arxiv.org/abs/2611.08946>
- **PDF：** <https://arxiv.org/pdf/2611.08946v1>（v1，2026-11-06）
- **项目页：** <https://flex-rsi.github.io>（页面可打开，仅核对到首页 HTML 的标题与描述，排行榜内容未逐项核对）
- **代码：** <https://github.com/flex-rsi/rsi-bench>（README 已读，未运行）
- **数据：** Hugging Face `Flex-RSI/RSI-Dataset`（README 称含 5 个任务的已整理切分；其余任务需 `rsib task prepare`）
- **机构（论文首页）：** Texas A&M University、University of Cambridge、National University of Singapore、NVIDIA、MIT、MIT-IBM Watson AI Lab、UT Austin、CMU、UPenn、USC、Purdue、Adobe、University of Toronto；三位组织者 Jingdong Zhang、Jionghao Wang、Zhengming Yu（Texas A&M，§7）
- **入库日期：** 2026-11-09
- **一句话说明：** 面向视觉与机器人任务的基准：在固定预算下，让前沿智能体在「改进任务解」与「改写自己的 harness（提示、工具、控制循环）」两层交替迭代，用隐藏测试集计分；24 任务、18 个模型。

## 核对范围与通读

- **核对范围：** 读 arXiv v1 PDF（pdftotext）正文 §1–§7 全文（含 §4.6 定性例子、Table 1–9、Figure 1–4 图注）；附录 A 全部 24 个任务卡（Action、Reasoning、Creation、Reconstruction、Perception、Infrastructure）；附录 B 全文（含 Table 10）；附录 C 的 Table 11–32 逐表读表头、说明与关键数字（未逐格重算）；附录 D 只读图注与每行的前后对比说明。未读：参考文献列表；图像本身的像素内容（只读图注和文字）；项目页排行榜内容（仅看到首页 HTML 标题与描述）。
- **通读：** 已完成

## 开源状态

- 仓库 `flex-rsi/rsi-bench` README 可访问，描述 `rsib` 命令行、任务表、数据下载与安装；README 称「提交运行结果进入排行榜的通道将尽快开放」。许可证文件在 main 分支下按 `LICENSE` 路径取到 404，**许可证未核实**。

## 核心摘录

1. **问题设定（§3.1）：** 每个任务给 starter 解、算力、预算与三个互不相交切分：开发集（带标签，智能体可用）、验证集（对智能体隐藏，决定候选解是否保留）、测试集（隐藏，运行结束后只用一次）。
2. **harness（§3.2）：** `H=(E,I)`，executor 改任务解，improver 改 executor 的提示/工具/工作流以及自己的提示、设置与经验；二者调用同一模型，模型权重不变。候选解仅当验证分超过已接受解加噪声边际 δ 才保留（式 2），起始 δ 取 starter 验证分标准差的 2 倍，improver 可改该规则（附录 B.4）。
3. **规模（§1、§4.1）：** 24 任务分六个领域（Action 7、Reasoning 7、Creation 4、Reconstruction 3、Perception 2、Infrastructure 1，按 Table 1），18 个模型（11 闭源、7 开源权重）。主结果只覆盖被至少 6 个模型评测的 20 个任务，另 4 个任务（Embodied-SWE、Spatial-VSI、Text-to-Image、VAE Decode）目前只有 2 个模型（Table 12）。
4. **总榜（§4.2、Figure 2）：** 分数为「该模型在各任务上击败其他模型的比例」再按领域平均，Claude Opus 5.5 91.6，Gemini 4 Argon 89.0（只跑了 9 个任务），GPT-6.1 Sol 77.0，GPT-6 Astra 65.5；最好的开源权重模型 Qwen3.8-Flash-Next 49.1 排第五。
5. **自我改进幅度（§4.3）：** 177 次运行中 150 次（85%）验证分提升，中位数 +10.4；闭源 94%（85 次运行，中位 +14.9），开源 76%（92 次，中位 +8.5）。
6. **三个发现（§5.1–§5.3）：** 自己构建解的任务提升空间大且拉开模型差距（9 个按量表打分任务上，模型起点相差 0.9 分、终点相差 38.8 分）；反馈只记成败时开源模型几乎不进步（4 个任务上开源中位增益 1.6，闭源 16.9；RoboCasa 上 5.6 对 51.2）；严格的物理稳定性或精确正确性检查处开源落后最多（Scene from Video 中开源模型的 Physics 分全为 0.0，Opus 70.7）。
7. **配对实验（§5.4、Table 8）：** 7 对「冻结 harness vs 自改进 harness」中 6 对测试分更高（+2.0 至 +40.0），1 对持平；Table 9 显示强模型的 harness 迁移给弱模型 4 个任务中仅 2 个有提升。
8. **成本（§4.5）：** 单次运行 token 中位数 49.7M（中间 80% 为 8.6M–93.4M）；12 个模型上相对 token 用量与得分的秩相关为 −0.05。
9. **作者声明的边界（§6）：** 模型权重不变，多数任务不训练模型；部分任务的解可能在开发集上训练或微调辅助模型；该基准不衡量模型修改自身参数。

## 对 wiki 的映射

- [paper-flex-rsi](../../wiki/entities/paper-flex-rsi.md)
- [递归自我改进（RSI）](../../wiki/concepts/recursive-self-improvement.md)

## 附录补充摘录（第二轮通读新增）

1. **评审器（附录 A.3）：** 创作类任务（Image Editing、Multi-turn Editing、Text-to-Image）由冻结的 GLM-4.5V 评审打分，并配程序化检查；Video Generation 用 T2V-CompBench 原评测器（LLaVA-v1.6-34B、Grounding DINO），且因子集与固定分辨率，分数与官方榜不可比。
2. **任务设计细节（附录 A.1、A.2）：** RoboDojo 与 RoboCasa 的 starter 都是「每个动作块调一次模型」的程序，运行中不训练策略；LIBERO-MAX/PRO 的 starter 直接执行冻结的 X-VLA 策略动作块、不调模型；ARC-AGI-3 测试集是 25 个公开游戏中按哈希分出的 10 个；Kinematics 与 Mass 只允许使用固定的开源 Qwen3.8-27B 作为语言模型，不调用被评测模型。
3. **逐任务结果表（附录 C，Table 13–32）里能看到最终测试分不高于 starter 的情形：** Video Quality 中 GPT-6 Astra 最终 82.1，starter 82.6（Table 32）；Kinematics 中 Grok 4.6 最终 37.2，starter 47.2（Table 20）；LIBERO-PRO 六个模型里只有 Claude Opus 5.5 高于 starter 的 53.3（Table 18）；Scene from Video 中三个开源模型（Qwen3.8-Flash-Next、MiniMax-M3、Qwen3.8-27B）验证分始终停在 17.3（Table 30）。
4. **ARC 开源模型（Table 22–24）：** ARC-AGI-2 上最佳开源 DeepSeek-V4.1 Flash 为 46.7（与其 starter 相同），ARC-AGI-3 上开源模型都在 1 以下；多数开源运行在 1–7 轮即被时间预算终止。
5. **完整性提示：** Table 12 注明 Text-to-Image 是任务的「预发布版本」，两个模型均为中等推理强度、各设置最多 25 轮；Muse Spark 1.3 通过内部接口访问，只列出不排名（Table 14、20、21）；GPT-6 Astra 最高推理强度的运行单独标出不排名（Table 29、30）。
6. **§4.6 与附录 D：** 前后对比的定性例子，如 RoboDojo Fill Egg Holder 由第 0 轮 0 分到第 3 轮 90 分（Opus 5.5，Figure 5）；这些是作者挑选的例子，不代表整体。
