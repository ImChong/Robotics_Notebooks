# Eureka（arXiv:2310.12931）

> 来源归档（paper）

- **标题：** Eureka: Human-Level Reward Design via Coding Large Language Models
- **类型：** paper
- **作者与机构：** Yecheng Jason Ma, William Liang, Guanzhi Wang, De-An Huang, Osbert Bastani, Dinesh Jayaraman, Yuke Zhu, Linxi "Jim" Fan, Anima Anandkumar；NVIDIA、UPenn、Caltech、UT Austin（首作者工作完成于 NVIDIA 实习期间，致谢节）
- **arXiv：** <https://arxiv.org/abs/2310.12931>（首次提交 2023-10；本次读 v2，2024-04-30）
- **PDF：** <https://arxiv.org/pdf/2310.12931>
- **发表：** ICLR 2024（PDF 页眉 “Published as a conference paper at ICLR 2024”）
- **项目页：** <https://eureka-research.github.io/>
- **代码：** <https://github.com/eureka-research/Eureka>（2026-11-12 已完成官方 README 与主运行源码核验；见下方增量核查）
- **入库日期：** 2026-11-08
- **一句话说明：** 把环境源码和任务描述交给 GPT-4，让它写可执行的奖励函数代码，用「采样一批、在 Isaac Gym 里各跑一次 PPO、把各奖励分量的训练曲线写成文字反馈、再让 LLM 改写最优者」的进化循环迭代；在 29 个仿真任务上与人类专家奖励对比。
- **核对范围：** 读了 arXiv v2 PDF（45 页）正文第 1–6 节全文；附录 A（全部提示词）、B（环境与适应度函数）、C（L2R 基线）、D（实现细节、笔转实验、人类初始化、人类反馈、算力）、E（MuJoCo Humanoid）、F（补充结果，含 Fig. 10–16 的文字与图注）、H（局限与讨论）逐段读；图 1 与图 4 看了渲染图读数，其余图只读了图注与正文描述。附录 G（奖励代码示例）读了各例标题、得分、文字说明与反思文本，具体奖励代码逐行未细读（它们是示例清单，不承载新的定量结论）。没读：参考文献逐条内容；项目页上的视频与奖励文本（只读了项目页文字）；未下载或运行任何代码。
- **通读：** 已完成

## 开源状态

- 论文 §1 写「承诺开源所有提示词、环境与生成的奖励函数」。
- **历史状态（2026-11-08）：** 初次抓取时官方仓库请求受限，当时未核实内部代码。项目页展示各环境最佳奖励文本。
- **本次补核（2026-11-12）：** 官方项目页有 arXiv、PDF、Code 三个直接入口；Code 指向 <https://github.com/eureka-research/Eureka>。已通过 GitHub 官方仓库读取 README、LICENSE 信息、`eureka/eureka.py`、`eureka/cfg/config.yaml`、`eureka/utils/create_task.py`、`eureka/utils/extract_task_code.py`、`eureka/utils/file_utils.py`、`eureka/utils/misc.py` 与 `isaacgymenvs/isaacgymenvs/train.py`。仓库主许可证 **MIT**，包含可运行奖励生成/训练入口；项目页及代码归档分别见 [站点归档](../sites/eureka-research.md) 和 [源码归档](../repos/eureka-research-eureka.md)。**未实际执行训练**。

## 2026-11-12 源码增量核查（与论文全文通读分开）

- **代码版本：** 官方仓库 `eureka-research/Eureka` 的 `main` 提交 `9eee42808b52d8abc14845d1547bfde886403ccc`；源码审阅范围及命令见 [仓库阅读记录](../repos/eureka-research-eureka.md)。
- **实际入口：** `eureka/eureka.py` 用 Hydra 构造对话提示，OpenAI `ChatCompletion.create` 采样候选，将 AST 解析出的奖励函数写入任务 `compute_reward`；子进程运行 `isaacgymenvs/isaacgymenvs/train.py`，使用 `rl_games` 训练并通过 TensorBoard 读取 `consecutive_successes`、`gt_reward`、各奖励项。
- **迭代与复评：** 按每轮任务指标选择最优候选形成反馈；全局最优代码用 `num_eval` 次不同 seed 重新训练后保存 `final_eval.npz`。中间候选不是每个都做多随机种子复评。
- **配置差异：** 官方 README 中说明的 `sample=16`、`iteration=5` 与仓库 `config.yaml` 当前实际默认 `sample=3`、`iteration=1` 不一致；复现应通过 CLI 显式指定，不能按说明文字推断启动默认值。
- **复现边界：** README 要求 Isaac Gym Preview 4、Python 3.8、OpenAI API Key；已提供 ShadowHandSpin checkpoint 演示入口。只检查代码与说明，未验证现代 Python/LLM API 的兼容性、训练结果或真机实验。

## 核心摘录（均出自论文，带章节）

1. **问题设定（§2）：** 奖励设计问题 RDP = ⟨M, R, A_M, F⟩，F 是只能通过对策略求值来查询的适应度函数（用真值奖励评估）。奖励生成问题：给定任务描述 l，输出奖励代码 R 使 F(A_M(R)) 最大。
2. **三个设计（§1、§3）：** 环境源码作上下文（§3.1，不含已有奖励代码）、进化搜索（§3.2）、奖励反思 reward reflection（§3.3）。
3. **搜索规模（§3.2）：** 每个环境 5 次独立运行，每次 5 轮迭代，每轮 K=16 个样本；仅 16 个样本就能保证首轮至少有一个可执行奖励（论文称对所有所考虑环境成立）。
4. **反思内容（§3.3、附录 A Prompt 2、附录 G.1）：** 记录各奖励分量与任务适应度在训练中间检查点的标量值（附录示例里每 300 个 epoch 一次，给出最大/均值/最小），以文字形式写入下一轮提示。
5. **主结果（摘要、§1、§4.3、图 4）：** 10 种机器人形态、29 个 IsaacGym 任务（9 个 Isaac 原环境加 Dexterity 基准 20 个双手任务）；Eureka 在 83% 的任务上超过人类专家奖励，平均归一化提升 52%；Isaac 全部任务持平或超过人类，Dexterity 20 个里 15 个持平或超过。图 4 读数：Dexterity 20 任务平均成功率 Eureka 55.30、L2R 24.40、Human 45.90、Sparse 10.40（读图）。
6. **评估口径（§4.2、附录 D）：** 基线 Human 为基准自带的人写奖励，Sparse 即适应度函数本身，L2R 为 Yu et al. 2023 的模板式方法（实现中拿到了人类奖励分量，作者称这对 L2R 有利）；所有最终奖励用同一份 PPO（rl_games）与同一组为人类奖励调好的超参，各跑 5 次，报告 10 个检查点上任务指标最大值的平均。中间候选奖励只跑 1 次 PPO（附录 D）。平均归一化提升在报告时把每任务得分截断到 [0, 3]（附录 D）。
7. **笔转（§4.3、图 7、附录 D.1）：** 在 Shadow Hand 环境里把方块换成笔，不改物理参数；先用 Eureka 奖励预训练「笔重定向到随机 SO(3) 目标」，再用同一奖励在一串预设路点上微调；论文称两阶段后能连续转多圈，从零训练和仅预训练的策略连一圈都转不完。结论是仿真中的。
8. **人类反馈（§4.4、图 8、图 9、附录 D.2、D.3、G.4）：** 以人类奖励作首轮初始化，Eureka（Human Init.）在所选任务上均优于 Eureka 与 Human；Humanoid 跑步通过人写文字反思（例如指出「在单脚跳」）得到 Eureka-HF，20 名用户里 15 人偏好它，前进速度 5.58 对 7.53（图 9）。
9. **消融（§4.3、附录 F）：** 去掉进化（32 样本一次性）弱于 2 轮 Eureka；去掉奖励反思平均归一化得分降 28.6%；换 GPT-3.5 性能下降但在多数 Isaac 任务仍持平或超人类；MuJoCo Humanoid 上 Eureka 7.68 对 Human 5.92（附录 E，表 4）。
10. **算力（附录 D.4）：** 单台 8×A100，每次 Eureka 运行不到一天；论文称 4 张 V100 也够。

## 论文自述的局限（附录 H）

- 主要在仿真评估；论文称有「附录 F 里的初步真机实验（真机跳跃，Sim2Real 管线）」，但本次读到的 v2 PDF 的附录 F 没有这部分内容（全文检索「hopping」「real-robot」只出现在附录 G.4 的人类反思文字与附录 H）。
- 需要存在且易于人类定义的任务适应度函数 F。
- 只在单一模拟器（IsaacGym）与固定 RL 算法（PPO）上实验，唯一例外是附录 E 的 MuJoCo Humanoid。

## 对 wiki 的映射

- [paper-eureka-llm-reward-design](../../wiki/entities/paper-eureka-llm-reward-design.md)
- 清单索引页（不含论文细节）：[painode-332-eurekanvidia](../../wiki/entities/painode-332-eurekanvidia.md)
