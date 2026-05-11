# dm_control

> 来源归档

- **标题：** dm_control（DeepMind / Google DeepMind 连续控制仿真与 RL 环境栈）
- **类型：** repo
- **来源：** Google DeepMind
- **链接：** https://github.com/google-deepmind/dm_control
- **配套论文（任务与 API 设计）：** [DeepMind Control Suite](https://arxiv.org/abs/1801.00690)（Tassa et al., 2018）
- **官方后续软件论文（引用格式见仓库 README）：** [Software Impacts, 2020](https://doi.org/10.1016/j.simpa.2020.100022)
- **入库日期：** 2026-05-11
- **一句话说明：** 基于 MuJoCo 的 Python 仿真与 RL 环境基础设施：`suite` 基准任务、`mujoco` 绑定、`viewer` / `mjcf` / `composer` / `locomotion` 等组件。
- **沉淀到 wiki：** 是 → [`wiki/entities/dm-control.md`](../wiki/entities/dm-control.md)

---

## 仓库定位（与论文关系）

- **arXiv:1801.00690** 系统描述了 **DeepMind Control Suite**：连续控制基准任务族、统一观测/动作/奖励约定、以及 `dm_control` 的 RL 环境与底层 MuJoCo Python 接口设计哲学。
- 当前 GitHub 仓库在论文之后演进为更完整的 **`dm_control` 包**：除 `suite` 外，还提供可组合环境的 `composer`、程序化 MJCF 的 `mjcf`、交互式 `viewer`、以及 `locomotion`（含足球等多智能体任务）等扩展模块；安装与渲染说明以 README 为准。

## 为什么值得保留

- 与 OpenAI Gym 连续控制域对标的 **标准化基准**，奖励大多落在 \([0,1]\)，便于横向对比算法与画学习曲线。
- **强可观测** 任务设计与 **物理稳定性** 验证流程（论文中强调用多种学习体迭代任务，避免「作弊」与数值爆炸）对自建仿真环境有方法论参考价值。
- 仍是 MuJoCo 生态中 **高引用** 的 Python 入口之一，大量研究与教学示例依赖 `suite.load(...)` 与 `TimeStep` API。

## 对 wiki 的映射

- [dm-control](../../wiki/entities/dm-control.md)
