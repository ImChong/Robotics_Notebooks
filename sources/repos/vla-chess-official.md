# VLA Adaptation Chess 官方源码核验
- **源码：** https://github.com/Kripner/vla-chess
- **许可：** Apache-2.0（`LICENSE` 与官方 README）
- **核对：** 2026-11-11
- **核对范围：** 官方 README 全文，`chess/sim/collect.py`、`chess/sim/eval_e2e.py`、`chess/train/finetune_real.py` 实际入口及功能，代码目录树。
- **通读：** 已完成

## 运行入口与模块边界
1. `python -m chess.sim.collect king_reach --n 10300 --seed 7000 --style reach --workers 16`：MuJoCo 的 `chess/sim/scene.py` + 脚本 expert 与 `truth.py` 分数，数据写 LeRobot episode，并标注 approach/pick/transport/place/home 五阶段。
2. `python -m chess.train.splits king_reach --sizes 100,316,1000,3162,10000`：对 success 和 index 产生固定 `splits/`。
3. `torchrun --nproc_per_node 8 -m chess.train.finetune_real king_reach --pretrained lerobot/pi05_base ...`：训练入口加载 LeRobot 数据与 `chess.real.segments` stage labels，`policy/core.py` 构造策略，`policy/pi05_subtask.py` 负责阶段预测，保存 checkpoint/control.json/subtask.json/provenance.json。
4. `python -m chess.sim.eval_e2e --ckpt ... --schedule schedules/hard_v1.json --k 1000 --envs 16`：脚本使用 `chess.sim.scene` 仿真、`policy` 产生分块动作和阶段切换，`chess.sim.truth` 统一打分，输出 `trials.csv` 与 `meta.json`。
5. 数据 https://huggingface.co/datasets/Kripi/so101-chess-king-moves-real（261 条，LeRobot v3，分段和 outcomes）。

## 工程注意
训练指定 CUDA/torchrun、EGL 等条件；真实机械臂仍需校准与设备安全控制，官方仓库存在代码不等于此处已复现。
- [论文](../papers/vla_adaptation_chess_arxiv_2611_12676.md) · [wiki](../../wiki/entities/paper-vla-adaptation-chess.md)
