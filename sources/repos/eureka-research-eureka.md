# Eureka 官方源码：eureka-research/Eureka

> 来源归档（repo / source-code reading）

- **英文标题：** Eureka: Human-Level Reward Design via Coding Large Language Models
- **中文标题：** Eureka：利用编程大语言模型实现人类水平的奖励函数设计
- **官方仓库：** <https://github.com/eureka-research/Eureka>
- **核对快照：** [`9eee42808b52d8abc14845d1547bfde886403ccc`](https://github.com/eureka-research/Eureka/tree/9eee42808b52d8abc14845d1547bfde886403ccc)（读取于 2026-11-12）
- **官方项目页：** <https://eureka-research.github.io/>
- **原始论文：** <https://arxiv.org/abs/2310.12931>（ICLR 2024）
- **代码许可证：** MIT（官方仓库 LICENSE 和 README；Isaac Gym 的依赖另行核对）
- **源代码范围：** 阅读 README 全文；`eureka/eureka.py` 主循环；`eureka/cfg/config.yaml`；`eureka/utils/{create_task,extract_task_code,file_utils,misc}.py`；`isaacgymenvs/isaacgymenvs/train.py` 的训练入口。没有逐行审阅随仓库附带的全部仿真任务和 rl_games 子模块；未执行命令。
- **入库日期：** 2026-11-12
- **沉淀到 Wiki：** [Eureka 论文实体页](../../wiki/entities/paper-eureka-llm-reward-design.md)

## 实际代码路径

1. **任务与 prompt：** `eureka/eureka.py:main(cfg)` 使用 Hydra 读取 `cfg.env.task`、`cfg.env.description`，加载 `eureka/envs/isaac|dexterity/` 的任务源码、`*_obs.py` 观测文本、`eureka/utils/prompts/*.txt`。
2. **生成候选：** 调用 `openai.ChatCompletion.create(model=..., messages=..., temperature=..., n=...)`。抽取模型输出中的 Python 函数代码，使用 `get_function_signature`（Python AST）解析接口。
3. **注入奖励：** 生成含 `self.rew_buf[:], self.rew_dict = ...` 的 `compute_reward` 调用；写入 `isaacgymenvs/isaacgymenvs/tasks/<env><suffix>.py`，必要时加 `@torch.jit.script`。通过 `create_task` 克隆并调整 task/train YAML。
4. **启动训练：** `set_freest_gpu()` 选择 GPU，`subprocess.Popen(['python','-u',ISAAC_ROOT_DIR+'/train.py', ...])` 启动 `train.py`。训练入口实例化 Isaac Gym 并通过 `rl_games.torch_runner.Runner.run({'train':True,...})` 开始策略学习。
5. **读取指标：** `block_until_training` 等待 stdout 出现训练或报错信号；主脚本从训练输出提取 `Tensorboard Directory`，`load_tensorboard_logs` 读标量。通常以 `consecutive_successes` 最大值挑选候选（缺失该指标时用 `gt_reward` 提供反馈），各奖励分量最大/均值/最小值被拼成下一轮的文字 `policy_feedback`。
6. **进化循环：** `np.argmax(successes)` 选本轮表现最高的候选；把最优奖励文本与反馈放入 `messages`，后续仅更新最近一次的 assistant/user 回合。保存 `summary.npz`、`messages.json`。
7. **最终复评：** 从全轮最佳代码复制回任务文件，按 `num_eval` 次运行训练（显式 `seed=i`）；写出 `final_eval.npz`。主流程未在机器人实机部署。

## 官方运行入口与配置差异

```bash
# 在 eureka/ 目录下；按 README 需先安装 Isaac Gym Preview 4 和依赖
python eureka.py env=shadow_hand sample=4 iteration=2 model=gpt-4-0314
# 官方提供的 ShadowHandSpin checkpoint 演示（在 isaacgymenvs/isaacgymenvs 下）
python train.py test=True headless=False force_render=True task=ShadowHandSpin checkpoint=checkpoints/EurekaPenSpinning.pth
```

| 参数 | README Getting Started 文字 | `eureka/cfg/config.yaml` 实际默认 |
|------|---------------------------|-----------------------------------|
| `sample` | 16 | **3** |
| `iteration` | 5 | **1** |
| `max_iterations` | — | 3000 |
| `num_eval` | — | 5 |

**复现注意：** 论文评测设置（5 次独立运行、每次 5 轮、每轮 K=16）与 README 示例和当前配置文件默认值是三种不同概念；应显式传参。官方 README 指定 Python 3.8、IsaacGym Preview 4、OpenAI API Key 和历史 `gpt-4-0314`；本次**未验证**该历史模型 API 在 2026 年仍可调用。动态执行 LLM 生成的 Python 奖励函数存在代码执行风险，建议仅在受控沙箱复现实验（工程建议）。

## 对 Wiki 的映射

- [Eureka：论文、Mermaid 主流程图与源码时序图](../../wiki/entities/paper-eureka-llm-reward-design.md)
- [奖励设计](../../wiki/concepts/reward-design.md)
- [Eureka 原文通读档案](../papers/eureka-llm-reward-design_arxiv_2310_12931.md)
- [官方项目页归档](../sites/eureka-research.md)
