# k-r-allen/residual-policy-learning（RPL 官方实现）

> 来源归档

- **标题：** Residual Policy Learning
- **类型：** repo
- **来源：** MIT CSAIL（Tom Silver / Kelsey Allen）
- **链接：** <https://github.com/k-r-allen/residual-policy-learning>
- **项目页：** <https://k-r-allen.github.io/residual-policy-learning/> — 归档见 [`sources/sites/residual-policy-learning-github-io.md`](../sites/residual-policy-learning-github-io.md)
- **入库日期：** 2026-07-28
- **一句话说明：** Silver et al. RPL（arXiv:1812.06298）官方代码：6 个 MuJoCo 操作环境（`rpl_environments`）+ 基于 OpenAI baselines DDPG/HER 的 TensorFlow 实验脚本。
- **沉淀到 wiki：** [`wiki/entities/paper-residual-policy-learning.md`](../../wiki/entities/paper-residual-policy-learning.md)

---

## 核心定位

RPL 论文官方仓库（约 83 stars，入库日核查）。包含两部分：

| 组件 | 路径 | 说明 |
|------|------|------|
| 环境包 | `rpl_environments/` | gym 注册环境：`SlipperyPush-v0`、`FetchHook-v0`、`TwoFrameHookNoisy-v0`、`ComplexHook-v0` 等；ComplexHook 需另下 718 MB 资产包（`fetch_complex_objects.zip`，MIT 托管） |
| 实验脚本 | `tensorflow/experiments/run_all_experiments.sh` | 复现论文全部实验的入口；依赖 OpenAI baselines（pin commit `c28acb2`） |

## 运行要点（README）

- **依赖：** Python 3.5.6（Ubuntu 18.04 / macOS 测试）；MuJoCo **mjpro 150** + mujoco-py（pin commit `a9f563c`）；baselines pin commit。
- **仅使用环境：** `pip install -e rpl_environments` 后 `gym.make("SlipperyPush-v0")` 等。
- **技术债提示：** TF1 + mujoco-py 150 时代栈，现代环境复现需适配（这是 2018 年代码，勿以当前依赖直接安装）。

## 与仓库内实体的关系

| 关联 | 说明 |
|------|------|
| [paper-residual-policy-learning](../../wiki/entities/paper-residual-policy-learning.md) | 本仓库对应的论文实体页 |
| [paper-residual-rl-robot-control](../../wiki/entities/paper-residual-rl-robot-control.md) | Johannink et al. Residual RL（ICRA 2019）：同期独立工作，论文正文互相引用；其真机代码未开源，本仓库是两篇共同思想的主要可运行入口 |
| [paper-reskill-residual-skill-policies](../../wiki/entities/paper-reskill-residual-skill-policies.md) | ReSkill 的 4 个下游任务直接改编自本仓库环境（Fetch 臂 + hook/push 族） |
