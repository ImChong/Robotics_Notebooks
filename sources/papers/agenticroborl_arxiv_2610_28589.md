# AgenticRoboRL: Bringing Coding-Agent Post-Training to Agentic Robotics（arXiv:2610.28589）

> 来源归档（paper / technical report）

- **标题：** AgenticRoboRL: Bringing Coding-Agent Post-Training to Agentic Robotics
- **arXiv：** <https://arxiv.org/abs/2610.28589>（v2，2026-10-21；[HTML](https://arxiv.org/html/2610.28589v2)、[PDF](https://arxiv.org/pdf/2610.28589)）
- **项目页：** <https://xinxinliu-ai.github.io/AgenticRoboRL/>
- **代码：** <https://github.com/xinxinliu-ai/AgenticRoboRL>（仓库已从用户给出的 L-CodingSpace 地址迁至 xinxinliu-ai）
- **模型：** <https://huggingface.co/ciciciciliu/AgenticRoboRL-Qwen3.5-9B>
- **作者：** Xinxin Liu, Bingxin Xu, Wenrui Bao, Ming Li, Wentao Bao, Chen Chen, Yuzhang Shang
- **机构：** University of Central Florida；University of Southern California；Michigan State University（按项目页和论文署名）
- **版本：** v2，2026-10-21，cs.RO
- **许可证：** 代码仓库 Apache-2.0（README 与仓库 license）
- **入库日期：** 2026-10-26
- **一句话说明：** 固定机器人 harness 的工具与执行控制，只后训练其语言模型 planner；通过教师成功轨迹 SFT 初始化，再用带成功结果奖励的多轮 RL 提升 planner 在不同 harness 中完成操作任务的能力。
- **对 wiki 的映射：** [AgenticRoboRL 实体页](../../wiki/entities/paper-agentic-roborl.md)

## 核心论文摘录

### 1. 共享接口放在模型调用边界

- **出处：** §3.1–3.2。
- **要点：** 不同机器人 harness 的执行输出可能是 VLA skill 调用，也可能是对感知/控制 API 的 Python 程序。框架不统一低层动作表示，而在 planner 请求—响应边界记录真实上下文、生成 token、rollout log-probabilities，以及适用时的图像和工具规范。
- **对 wiki 的映射：** 适配器连接训练数据和任务结果，不会替换或训练 harness 的执行模块。

### 2. 两阶段 planner 后训练

- **出处：** §3.3。
- **SFT：** 从更强 teacher 的成功轨迹构建学生实际会收到的逐轮上下文；RPent 中 teacher 有外部记忆而 student 没有，因此重建 student 上下文，并筛查/改写依赖 teacher 私有记忆的回答。
- **RL：** 在同一任务与初始状态的有效尝试组内计算 leave-one-out advantage；使用终局成功奖励，不训练 critic，也不做奖励标准差归一化。每个 episode 最多采样 16 个合格 planner response，以 dual-clip PPO token loss 更新 planner。
- **有效性检查：** adapter 返回任务、初始状态、attempt、reward 和 validity；任务失败仍是零奖励样本，基础设施故障样本从优化中排除。训练前校验请求记录完整、policy version 一致及重新渲染 prompt 的 token 对齐。

### 3. 结果与边界

- **出处：** §4.2–4.5；项目页 Results。
- **RPent / LIBERO-PRO：** Qwen3.5-9B 基线 18.1%，SFT 后 65.1%，SFT+RL 后 71.0%；最终评测为 8 个 suite、共 800 episodes。
- **泛化测试：** 同一组 RPent checkpoint 在未额外训练的 LIBERO-Plus 10,030 个配置上，基线 19.2%、SFT 66.7%、SFT+RL 72.2%。
- **跨 harness：** CaP-X / LIBERO-PRO 使用生成控制代码的 planner，基线 2.7%、SFT 18.7%、SFT+RL 28.5%；该设置使用 privileged state，数据与 RPent 实验不同。
- **比较边界：** 论文将 RPent 最终 71.0% 与使用外部记忆的 Codex (GPT-5.5) 72.1% 对照；二者模型规模、记忆条件和训练方式不同，适合作为结果参照，不是同条件模型比较。

## 官方资料核查

- 项目页提供论文、代码和 Hugging Face checkpoint 入口。
- 仓库公开 sft/、rl/、RPent 与 CaP-X harness 配置、adapter、toy minimal harness、评测/训练控制器和测试目录；README 给出可执行训练命令。
- Apache-2.0 仓库包含可运行代码；已发布 Qwen3.5-9B 的 SFT 与 RL checkpoint。
- **复现限制：** README 说明 teacher 轨迹未发布。完整 RPent / CaP-X 复现还依赖各自的 harness、模拟器环境、VLA checkpoint（RPent）以及 vLLM / PyTorch 训练环境；仓库另提供不需 GPU、模拟器或网络的 minimal 流程用于检查整体训练回路。
- **对 wiki 的映射：** 工程实践和局限。

## 对 wiki 的映射

- 统一实体：[wiki/entities/paper-agentic-roborl.md](../../wiki/entities/paper-agentic-roborl.md)
- 项目页归档：[sources/sites/agenticroborl-project.md](../sites/agenticroborl-project.md)
- 官方仓库归档：[sources/repos/agenticroborl.md](../repos/agenticroborl.md)
