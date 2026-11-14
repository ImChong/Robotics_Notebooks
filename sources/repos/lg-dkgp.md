# LG-DKGP：Rethinking Actions 官方复现代码

> 来源归档（repo） · https://github.com/oliverstack-byte/LG-DKGP

- **类型：** repo
- **项目：** LG-DKGP（Locality-Gated Deep-Kernel Gaussian-Process Prior）
- **论文：** [Rethinking Actions（arXiv:2611.16645）](../papers/lg_dkgp_arxiv_2611_16645.md)
- **代码：** https://github.com/oliverstack-byte/LG-DKGP
- **模型：** https://huggingface.co/yangbobome/LG-DKGP （README 发布链接）
- **许可：** MIT；上游与第三方组件各保留自己的许可证
- **入库日期：** 2026-11-14
- **核查范围：** README、pyproject.toml、docs/data.md、src/lgdkgp/__main__.py、cli.py、data/prepare.py、training/prior.py、models/prior/lg_dkgp.py、planning/cem.py、evaluation/{runner,native,pc}.py。
- **开放状态：** **训练/评测代码已开源**，配方、清单、校验信息随仓库；原始训练数据、预训练权重和部分原生仿真 fixtures/scene assets **另行下载**，非仓库内直接提供。未在当前环境启动训练或模拟器。

## 官方命令（README，仓库根目录）

各任务依赖互斥；不要使用 --all-extras。

```bash
uv sync --locked --extra pc  # PushT / Cube；其他任务单独选 libero 或 robotwin
export CUDA_VISIBLE_DEVICES=0
# 以下以 PushT 为例；其余任务更换 configs/tasks/*.json
uv run --locked --no-sync python -m lgdkgp prepare --config configs/tasks/pusht.json
uv run --locked --no-sync python -m lgdkgp train --config configs/tasks/pusht.json --stage mean
uv run --locked --no-sync python -m lgdkgp train --config configs/tasks/pusht.json --stage gp
uv run --locked --no-sync python -m lgdkgp eval --config configs/tasks/pusht.json
```

独立完整复现可在 prepare 前用同一配置执行 train-world（训练 LeWM）并显式指定生成的 world_model；直接评测必须提前放好官方发布且**配对一致**的 prior.pt、world_model.pt、config.json 及 fixtures。

## 源码入口与运行时数据流

| 仓库路径 | 实际职责 |
|---|---|
| src/lgdkgp/__main__.py → cli.py::main | python -m lgdkgp 的真实入口；分发 prepare/train/train-world/eval/check |
| data/prepare.py::prepare | 载入原始数据与冻结 LeWM，提取视觉隐特征并准备按 episode 分开的历史/未来动作窗口 |
| training/prior.py::train | 先 mean（MSE），后 gp（联合变分目标），读取对应均值检查点并输出最佳先验 |
| models/prior/lg_dkgp.py::LGDKGP | 门控历史编码器、神经均值、深核 GP 残差及预测均值/协方差 |
| planning/cem.py::Planner.solve / cem | 当前/目标图像→冻结世界编码，历史动作→先验均值；围绕中心采样 CEM 候选，调用 world.get_cost 评分 |
| evaluation/runner.py::evaluate | 读取固定 queries、校验 checkpoint 和查询哈希，分发单视图 / native 任务，汇总每种子结果 |
| evaluation/pc.py::run | PushT / Cube 运行接口 |
| evaluation/native.py::run_case | LIBERO / RoboTwin 还原 fixture 和 prefix、观测、求解、执行动作、记录局部/终态成功 |
| configs/tasks/*.json | 五任务的先验/LeWM 检查点、执行预算、探索尺度和输出配置 |

关键真实时序（核对自源码）：evaluation/runner.evaluate → pc.run **或** 新进程 native.run_case → Planner.solve → LGDKGP.predict_distribution → cem → world.get_cost → env.step → Planner.observe(实际已执行动作) → 下一次规划。不是 GP 直接输出最终控制动作，也不是用未来专家动作作为在线输入。

## 工程与复现边界

- 主设置 H=P=25、budget=50、CEM 300×30、elite=30；只有世界模型用于候选目标评分。按任务加载释放的配对权重、动作归一化和冻结 query manifest。
- 训练可分 mean 与 gp 阶段；评测要求完成的 gp-stage prior，不接受 mean.pt 或未完成的 .last 中间状态。
- PushT/Cube 需 LeWM HDF5 数据；Caddy/Scene2/RoboTwin 需 SAGE 数据及独立 simulator 资源。训练资源较大，README/ docs/data.md 列明外部地址。
- 源码报告 evaluation/summary.json、binding.json（checkpoint / query SHA256）及逐 query 结果；静态核查并不等同于 GPU/模拟器实测。

## 对 wiki 的映射

- [论文节点：LG-DKGP](../../wiki/entities/paper-lg-dkgp.md)
- [论文原文归档](../papers/lg_dkgp_arxiv_2611_16645.md)
