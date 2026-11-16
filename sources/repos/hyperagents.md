# facebookresearch/Hyperagents｜Hyperagents 官方实现

> 原始源码归档（2026-11-16 核查）｜与 [arXiv:2603.19461v2](https://arxiv.org/abs/2603.19461v2) 一一对应

- **仓库：** <https://github.com/facebookresearch/Hyperagents>
- **README：** <https://github.com/facebookresearch/Hyperagents/blob/main/README.md>
- **许可证：** [LICENSE.md](https://github.com/facebookresearch/Hyperagents/blob/main/LICENSE.md) — **CC BY-NC-SA 4.0**，限制商业用途
- **开放状态：** **已开源**。公开包含元 Agent、任务 Agent、演化循环、Docker 与任务评测实现；论文的实验成绩不是本次独立复现实测。
- **核对范围：** README、`generate_loop.py`、`run_meta_agent.py`、`meta_agent.py`、`task_agent.py`、`domains/harness.py`、`domains/report.py`、`domains/genesis/eval.py`、`domains/genesis/evaluator.py`、`domains/genesis/README.md`、`utils/domain_utils.py`、`LICENSE.md` 的关键入口与相关函数。
- **通读：** 已完成
- **论文归档：** [Hyperagents v2](../papers/hyperagents_arxiv_2603_19461v2.md)
- **Wiki 节点：** [Hyperagents / DGM-H](../../wiki/entities/paper-hyperagents.md)

## 可执行入口（以官方 README 和源码为准）

1. README 使用 Python 3.12 虚拟环境、`requirements.txt` 与 `requirements_dev.txt`，需要设置模型服务 API Key，并构建容器：`docker build --network=host -t hyperagents .`
2. 初始化入口：`bash ./setup_initial.sh`；具体领域训练和评测依赖需单独准备，例如 Genesis 的 PyTorch、Genesis、RSL-RL、CUDA/GPU 环境。
3. 主演化入口：`python generate_loop.py --domains paper_review`，或指定 `--domains genesis_go2walking`；`generate_loop.py` 的参数包括 `--max_generation`、`--eval_samples`、`--parent_selection` 与 `--eval_test`。
4. Go2 初始奖励评测可按 `domains/genesis/README.md` 先执行 `python -m domains.harness --domain genesis_go2walking --run_id initial_genesis_go2walking_0 --num_samples 1 --num_workers 1`，再执行 `python -m domains.report --domain genesis_go2walking --dname ./outputs/initial_genesis_go2walking_0`。

## 静态核对的核心模块与数据流

| 实现位置 | 实际职责 |
|---|---|
| `generate_loop.py::generate_loop` | 创建 `docker.DockerClient`，初始化 `archive.jsonl`，调用亲本选择和代际迭代 `generate()` |
| `generate_loop.py::generate` | 在 Docker 中应用祖先 diff，启动 `run_meta_agent.py`，收集 `model_patch.diff`，执行编译检查、阶段/完整评测并保存 `metadata.json` |
| `run_meta_agent.py` → `meta_agent.py::MetaAgent.forward` | 从亲本加载既有代码与历史评测；调用 `agent.llm_withtools.chat_with_agent` 编辑可变程序，利用 Git diff 形成补丁；`domains/` 路径会被重置到初始 commit，保护域评测实现 |
| `task_agent.py::TaskAgent.forward` | 为当前任务组织输入，调用模型工具/聊天接口，按 JSON `response` 提取候选 |
| `domains/harness.py` / `domains/report.py` | 按领域装载任务 Agent、评测样本与指标；Genesis 分支转至 `domains.genesis.eval` |
| `domains/genesis/eval.py` / `domains/genesis/evaluator.py` | `EvaluatorManager` 分配 Genesis 任务，`Evaluator.run_episode` 将环境路径、默认奖励及任务描述交给 Task Agent；保存生成的 `reward_function_*.py`，调用 `domains.genesis.genesis_train.rl_trainer` 训练并运行 RL evaluation |
| `utils/domain_utils.py` | 规定 Genesis 评测指标为 `average_fitness`、阶段样本数 3、主要 split 为 train；不应将此当作独立未见测试集 |

## Go2 奖励设计的运行层级

**元层** `MetaAgent` 改写未来的任务代理或自改过程；**任务层** `TaskAgent` 编写候选 Python 奖励函数；**训练层** 用生成奖励进行 Genesis/RSL-RL 的 PPO 训练；**评估层** 返回仿真 fitness；**档案层** 保留演化日志、代码补丁与分数。与直接产生 50/1000 Hz 机器人控制指令不同。

论文报告的仿真 Go2 测试 `0.060→0.372` 是作者特定任务和计算预算下的结果；本次未执行 CUDA 环境、RL 重训或真机验证。

## 工程约束与安全

- Docker 隔离不是绝对保证；官方 README 明确警示模型生成代码可能具有破坏性。
- `domains/` 被复原，主实验的外部评测接口不在元层 Agent 可任意修改范围。
- 依赖外部模型 API、Docker、Genesis、RSL-RL 与 GPU；首次复现应先运行一个小规模配置，并固定种子、模型和评测条件。
- **许可：CC BY-NC-SA 4.0**；任何商业工程集成需单独核实授权条款。
