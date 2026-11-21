# TacRefineNet（NoneJou072/tacrefinenet）

> 来源归档（官方源码仓 · MIT / CC BY 4.0）

- **标题：** TacRefineNet — 官方代码仓库
- **类型：** repo
- **组织：** 小米机器人实验室（Xiaomi Robotics）相关作者；GitHub 账号 `NoneJou072`
- **代码：** <https://github.com/NoneJou072/tacrefinenet>
- **项目页：** <https://sites.google.com/view/tacrefinenet>
- **论文：** <https://arxiv.org/abs/2509.25746>（[`sources/papers/tacrefinenet_arxiv_2509_25746.md`](../papers/tacrefinenet_arxiv_2509_25746.md)）
- **入库日期：** 2026-07-23
- **核对范围：** 核对 `NoneJou072/tacrefinenet@8cabb33`（2026-08-07）仓库结构、`README.md`、`mivt/scripts/train.py`、`mivt/scripts/deploy_sim.py`、`mivt/util/deployer.py`、`Dockerfile`、`requirements.txt`。未运行 MuJoCo 仿真或 DDP 训练。
- **通读：** 已完成
- **一句话说明：** TacRefineNet 官方可运行代码仓库，提供基于 PyTorch DDP 的触觉抓取精修训练入口、MuJoCo 仿真闭环部署评测脚本以及 Google Drive 预训练权重/部分物体模型下载链接。

## 开源状态（2026-11-22 复核）

| 项 | 结论 |
|----|------|
| 仓库状态 | **已开源**（`8cabb33`，2026-08-07 更新，MIT LICENSE / README 标注 CC BY 4.0） |
| 训练入口 | `torchrun ... mivt/scripts/train.py --config train/config_server_tac.yaml --seed 42`（`ObjectGroupedBatchSampler` + `VTANet` / `PoseRegressionNet` DDP 训练） |
| 仿真部署评测 | `python -m mivt.scripts.deploy_sim --sim-config sim/config_local_tac.yaml --train-config train/config_local_tac.yaml -c assets/checkpoints/tacrefinenet-42.pth -o carlogo --headless`（支持 `--rnd-goal`、Oracle 终止、保守收敛停机与多进程 `deploy_sim_mp.py`） |
| 推理封装 | `mivt/util/deployer.py`（`Deployer.set_goal` / `Deployer.inference`：触觉与本体归一化、SVD 对称正交化、可选 `profile_latency` 统计） |
| 数据与权重 | README 附 Google Drive `assets/`（预训练权重、处理后数据集）；因商业保密**仅公开部分物体模型** |

## 对 wiki 的映射

- [TacRefineNet（论文实体）](../../wiki/entities/paper-tacrefinenet-tactile-grasp-refinement.md)
