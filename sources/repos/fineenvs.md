# FineEnvs

> 来源归档

- **标题：** FineEnvs — Open Source RL Environments for LLM Agents
- **类型：** repo / 开源环境、训练与评测配方
- **作者：** Adithya S. Kolavi（按仓库 Citation）
- **主链接：** <https://github.com/adithya-s-k/FineEnvs>
- **官方资源：** [FineEnvs Hugging Face 组织](https://huggingface.co/FineEnvs) · [RL Environments Guide](https://huggingface.co/spaces/AdithyaSK/rl-environments-guide)
- **许可证：** 仓库根目录声明 Apache-2.0；指南内容目录另有 CC-BY-4.0 声明。Hub 上每个数据集、模型和 Space 应分别核对自身许可证。
- **源码状态：** 已开源；公开仓库包含环境实现、训练脚本、notebook、结果记录、工具和 agent skills。
- **入库日期：** 2026-10-08
- **一句话说明：** 将 LLM 强化学习环境的设计、框架适配、rollout、训练、部署与评测连成可复现配方；覆盖游戏、代码执行、桌面操作、数据分析、视觉定位与港口调度任务。
- **沉淀到 wiki：** 是 → [FineEnvs 实体页](../../wiki/entities/fineenvs.md)

---

## 仓库内容核查

README 将每个编号目录定义为可独立运行的端到端项目，并将 recipes / notebooks / source 放在 GitHub，环境、数据集、模型与演示发布到 Hugging Face。首页列出：

- 00-environments-101：Jupyter agent、Wordle、Desktop 三类环境，在 OpenEnv、ORS、NeMo Gym、Verifiers、SkyRL Gym、GEM 六套框架中横向实现。
- 01-latex-ocr：通过 OpenEnv reward server 为视觉语言模型训练公式转 LaTeX 环境，配 notebook、GRPO runner 与 smoke tests。
- 02-watercolour、03-geoguesser：基于审美反馈绘画，以及多轮视觉地理定位任务。
- 04-smoldataenvs、05-multi-harness-rl：数据分析任务集与多-harness 训练 recipe。
- 07-simulation-environments/portsim-v1：以巴塞罗那港 2024 记录构建的泊位调度规划环境，README 描述其用 CP-SAT 求解的最优值评分。

## 工程要点

仓库的 RL 环境设计指南建议先确定任务、工具/动作、观测、执行后端、持久状态、奖励和终止条件，再选框架。部署边界（进程内或 HTTP）、交互轮数、奖励归属与状态隔离决定实现复杂度。对多轮任务，应先人工检查少量 rollout 轨迹和奖励触发，再扩大并行或启动训练。

最小本地示例见 Wordle + Verifiers README：纯 Python Wordle toolkit 暴露 guess / get_history 工具，rollout 脚本调用模型并逐轮交互；不需要 E2B sandbox。更重的桌面、代码执行环境则需要相应外部服务凭证。

## 使用入口

~~~bash
git clone https://github.com/adithya-s-k/FineEnvs
cd FineEnvs/00-environments-101/envs/wordle/verifiers
uv sync
uv run python rollout.py
~~~

该示例使用 HF Inference Providers，需要按 README 配置 HF_TOKEN；替换为 OpenAI 模型时才需要 OPENAI_API_KEY。完整 GPU 训练不是上述轻量 rollout 的必要条件。

## 交叉索引

- [FineEnvs 实体页](../../wiki/entities/fineenvs.md)
- [Gymnasium](../../wiki/entities/gymnasium.md) — 多框架集合中包含采用 Gymnasium 风格接口的 GEM；FineEnvs 本身不是 Gymnasium 的替代品。
- [Reinforcement Learning](../../wiki/methods/reinforcement-learning.md) — 环境接口、采样与策略训练的通用背景
- [指南源码 README](../sites/fineenvs-rl-environments-guide.md) — 公开指南的发布与内容源说明
