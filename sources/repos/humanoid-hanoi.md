# osudrl/Humanoid_Hanoi（官方环境代码）

- **类型：** repo / benchmark environment
- **代码：** <https://github.com/osudrl/Humanoid_Hanoi>
- **项目：** <https://osudrl.github.io/Humanoid_Hanoi/>
- **论文：** <https://arxiv.org/html/2602.13850>
- **核对范围：** GitHub main README.md、setup.bash、eval.py、util/env_factory.py、util/evaluation_factory.py、env/tasks/boxtowerofhanoienv/boxtowerofhanoienv.py 的构造、reset、step、load_plan 部分；核对了 env/sim/util 目录。非整个项目逐文件执行测试。
- **通读：** 已完成
- **核对日期：** 2026-11-22

## 实际公开内容与入口

- `bash setup.bash` 安装 Conda 环境以及 MuJoCo 2.1.0，脚本会交互询问创建或升级环境。
- `python eval.py --robot digit` 是 README 的入口，另支持 `--robot h1`、`--robot g1`、`--offscreen`；`eval.py` 用 `add_env_parser` 组装默认参数，调用 `env_factory("BoxTowerOfHanoiEnv", env_args)()`。
- `util/env_factory.py` 按环境名导入 `env.tasks.boxtowerofhanoienv.boxtowerofhanoienv.BoxTowerOfHanoiEnv`。
- `util/evaluation_factory.py::interactive_eval()` 启动 viewer、调用 `env.reset()`，循环运行 `env.step()` + `env.sim.viewer_render()`；键盘 `r` 重新调用 reset。
- `BoxTowerOfHanoiEnv.reset()` 依次重置仿真、随机化塔位与箱体尺寸/质量/摩擦、重设箱体状态；`load_plan()` 读取 `box-world-json/hanoi.json` 并组装动作计划。
- **重要：** 当前 `BoxTowerOfHanoiEnv.step()` 主要画标记，且仅在最初两帧调用零关节指令的 `step_simulation`；**未接入论文的四技能 50 Hz 策略、MHC 推理或训练逻辑**。运行交互窗口不等于复现论文的 47% benchmark。

## 开源边界与使用风险

官方 README 原文明确：**“This repository contains only the environment implementation. Training code is NOT included.”** 因此该仓库只能作为基准场景/模拟器参考，不包含论文报告的四技能 PPO 训练、WBC 参考数据聚合、已训练权重和 Digit V3 部署代码。代码中 `--offscreen` 参数传入交互例程，但后者仍调用 viewer；实际无显示环境运行效果本次未执行确认。

## 关联

- [论文一手归档](../papers/humanoid_pnb_humanoid-hanoi.md)
- [官方项目主页归档](../sites/humanoid-hanoi.md)
- [知识实体（含源码真实序列图）](../../wiki/entities/paper-notebook-humanoid-hanoi-investigating-shared-whole-body-c.md)
