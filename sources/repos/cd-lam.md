# CD-LAM 固定版本实现核读

- **URL：** <https://github.com/AetherLabsAI/CD-LAM>
- **版本：** [5540960839b03b56f96f190e85a316aeeb071406](https://github.com/AetherLabsAI/CD-LAM/tree/5540960839b03b56f96f190e85a316aeeb071406)
- **核对范围：** 根README、run.sh与scripts/run.sh、cdlam_runtime/cli.py、entries/stage1.py全文；runtime.py的commands_for_stage/execute；objectives.py三项核心损失完整函数及cd_lam/training/runner.execute_stage。2026-11-13静态核读，未执行GPU。
- **通读：** 已完成

真实stage/pipeline走cdlam_runtime，stage1→bridge→stage2→stage3。每阶段doctor检查、命令构造、subprocess执行；Stage1 adapter加载外部LAM、shard帧源及mask监督后进入外部loop。plan-stage与synthetic smoke走另一边界，不把smoke通过当真实训练。

objectives.py的前景权默认5/1，而论文1/.02；SigLIP helper对selected pair统一mean，论文式9分正负归一；不能把通用helper等同外部训练loop的完整参数/权重。2B权重/adapter与checkpoint专属22D→32D bridge公开，14B权重和运行adapter不含；Apache2.0仓库与外部模型许可分开核对。
