# WAM-Policy 官方训练、评测和部署仓库
- **仓库：** https://github.com/AIGeeksGroup/WAM-Policy
- **论文：** https://arxiv.org/abs/2611.24011
- **项目：** https://aigeeksgroup.github.io/WAM-Policy/
- **许可：** 主分支 Apache-2.0；deploy/ 与 qlm-bench 特别遵守 CC BY-NC 4.0
- **核对范围：** README、scripts/stage1_2_opsd/train.sh、scripts/evaluation/standard100.sh、deploy/README.md；并对照 Stage1/Stage2 脚本入口。
- **通读：** 已完成
- **入库日期：** 2026-11-23

## 源码入口
- Cross-morphology transfer：scripts/stage1_1_transfer/convert_base.sh、prepare_data.sh、train.sh
- OPSD：scripts/stage1_2_opsd/train.sh → python -m wam_policy.opsd train
- RL specialists：scripts/stage2_1_task_rl/run_task_rl.sh
- M-OPD：scripts/stage2_2_mopd/train.sh
- 标准评测：scripts/evaluation/standard100.sh → python -m wam_policy.evaluation.standard100
- Go2 真机：deploy/tasks/push_box.sh 与 inference service / ROS2 RAMBO

详见[论文详情](../../wiki/entities/paper-wam-policy.md)。未实际运行训练、评测或真机。
