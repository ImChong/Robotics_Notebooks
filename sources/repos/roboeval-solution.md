# IliaLarchenko/RoboEvalSolution 官方源码

- **地址：** https://github.com/IliaLarchenko/RoboEvalSolution
- **核对提交：** `05b530d8c6e9686002136f94241d428d90ef78ca`；子模块 RoboEval `c8b79598c9fb9021652128020517d1847abe4c95`、openpi `c23745b5ad24e98f66967ea795a07b2588ed6c79`。
- **论文：** https://arxiv.org/abs/2611.11464
- **项目：** https://ilialarchenko.com/projects/roboeval2026/
- **权重：** https://huggingface.co/IliaLarchenko/RoboEvalSolution
- **许可：** Apache-2.0（子模块各自许可）
- **核对日期：** 2026-11-25
- **核对范围：** README、scripts/rl_loop.py 的 trainer/collector 入口与调用、scripts/serve_policy.py 的 BatchedPolicy 调度和选择、scripts/rollout.py 的 Slot 请求/响应/环境步进/官方评分、ParamTuner 以及 configs/settings.json；静态阅读，不是执行 trace。
- **通读：** 已完成

## 运行入口和模块

| 模块 | 作用 |
|---|---|
| scripts/rl_loop.py | train_round 构建数据池、compute_advantages；collect 启动 server 和 rollout.py --record；recover 使用 --recover |
| scripts/bucket_daemon.py | README 指定的 trainer/collector 数据与 checkpoint 同步进程 |
| scripts/serve_policy.py | WebSocket batched server，transform_one、launch、finish_batched、_select |
| scripts/rollout.py | Slot._request 发送图像、本体状态、task id、inpaint 和 overrides；on_response 重采样；step 执行；_finish 计算官方分数 |
| src/roboeval_solution/data/score_reward.py | README 指定的 RDS 奖励转换 |
| src/roboeval_solution/training/flywheel.py | README 与 rl_loop 引用的数据更新与优势计算 |
| configs/settings.json | 最终逐任务推理配置 |

## 训练采集与固定评测的运行分岔

`rl_loop.py` 将 `--param-search` 传给 `rollout.py`；协调器构造 `ParamTuner`，用 `top_up` 为各任务补样本；worker 在回合开始领取，并在回合结束后回传 `score_ep`，由 `update` 排除错误回合并消费分数（不会从回合中途的单帧更新后验）。换新 run 或搜索空间时，按策略步数对已有记录加新近权重重拟合。固定评测的 `--best` 只由任务读取配置，分发服务端采样覆盖项和客户端执行参数；仅在另传 `--param-search` 时才构造协调器搜索器。

## 官方评测入口

```bash
git clone --recurse-submodules https://github.com/IliaLarchenko/RoboEvalSolution.git
cd RoboEvalSolution
bash setup.sh
.venv/bin/hf download IliaLarchenko/RoboEvalSolution --local-dir ../checkpoint
CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/serve_policy.py --config-name pi05_roboeval --checkpoint-dir ../checkpoint --num-candidates 6 &
MUJOCO_GL=egl .venv-sim/bin/python scripts/rollout.py --server ws://localhost:8765 --out ../runs/eval_x --episodes-per-task 50 --workers 20 --best configs/settings.json
```

README 提供 Linux/NVIDIA CUDA 13 或 --cuda12 双环境；训练需额外公共示范（约 96 GB）、存储 bucket。上述 rollout 拒绝正式公布的 1000–1049 seeds，输出 summary.json，不能把它称为主办方最终复评。权重和实际依赖未下载验证。

## 对 wiki 的映射

- [论文实体](../../wiki/entities/paper-roboeval-solution.md)
- [论文归档](../papers/roboeval_solution_arxiv_2611_11464.md)
- [项目页归档](../sites/roboeval-solution.md)
