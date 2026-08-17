# sonic-x2（GEAR-SONIC → AgiBot X2 Ultra MuJoCo play bundle）

> 来源归档

- **标题：** sonic-x2
- **类型：** repo
- **来源：** Sitarama Chekuri / meetsitaram
- **链接：** <https://github.com/meetsitaram/sonic-x2>
- **项目页：** <https://sonic-agibot-x2.github.io/sonic-transfer/>
- **权重卡：** <https://huggingface.co/tinkerbuggy/sonic-x2>
- **许可：** 截至 2026-08-17 GitHub **未挂 SPDX LICENSE**
- **入库日期：** 2026-08-17
- **一句话说明：** 在 MuJoCo 里跑 AgiBot X2 Ultra（31 DoF）上的 SONIC 全身跟踪：ONNX 策略、MJCF、参考动作与 deploy-parity 调参；**默认模型是冻结 G1-core + LoRA transfer（v2）**。
- **沉淀到 wiki：** [`wiki/entities/paper-sonic-transfer.md`](../../wiki/entities/paper-sonic-transfer.md)

---

## 核心定位

论文页的 **可运行入口**，不是完整训练仓。CPU-only：`./install.sh` 建 venv（mujoco / onnxruntime / scipy / joblib）并拉 mesh，然后 `./play_v2.sh`。

完整 gamepad / 遥操作 / 规划器 / 真机 bring-up 指向 sibling [`meetsitaram/GR00T-WholeBodyControl-X2-review`](https://github.com/meetsitaram/GR00T-WholeBodyControl-X2-review)（本条 ingest 不展开）。

## 开源边界（2026-08-17）

| 项 | 状态 |
|----|------|
| MuJoCo ONNX 回放 | **可运行**（`scripts/eval_x2_mujoco_onnx.py`，50 Hz） |
| Transfer 权重 | `models/x2_sonic_frozen_g1core_lora_v2.onnx`（phase-3 8900；README 写 PHUMA 69.0 vs 59.0） |
| Codec sidecar | `.phi.json` 与 v2 模型成对 |
| Incumbent 对照 | `models/x2_sonic_14000_g1.onnx`（原生 X2 14k） |
| LoRA 训练 / Isaac Lab | **不在本仓** |
| LICENSE | **未声明**；复用前需核对授权 |

## 仓库入口（README）

| 路径 / 命令 | 说明 |
|-------------|------|
| `./install.sh` | `.venv` + mesh |
| `./play_v2.sh [gangam\|walk\|idle]` | **默认 transfer v2**：`--tuning '' --action-clip 20 --freeze-wrist` |
| `./play_relaxed_walk.sh` 等 | 可 `MODEL=...` 切 incumbent；incumbent 用 `configs/real_deploy_tuning/bigrun.yaml` |
| `models/x2_sonic_frozen_g1core_lora_v2.onnx` | 冻结 G1-core + LoRA；1670-D obs → 31-D action |
| `models/x2_sonic_14000_g1.onnx` | 原生 incumbent |
| `motions/*.pkl` | joblib dict；`dof` 31 关节 MJCF 序；走/舞 50 fps，idle 30 fps |
| `assets/mjcf/x2_ultra.xml` | X2 Ultra MuJoCo |
| `scripts/eval_x2_mujoco_onnx.py` | ONNX 玩家：deploy-tuned PD、RSI、跌倒检测、跟踪指标 |

**调参陷阱：** incumbent 的 `bigrun` deviation clamp 会 **搞崩** v2；transfer 必须用 parity gains + action clip + 冻腕。

## 与仓库内实体的关系

| 关联 | 说明 |
|------|------|
| [paper-sonic-transfer](../../wiki/entities/paper-sonic-transfer.md) | 冻结 codec + LoRA 论文实体 |
| [SONIC](../../wiki/methods/sonic-motion-tracking.md) | 源平台 GEAR-SONIC |
| [GR00T-WholeBodyControl](../../wiki/entities/gr00t-wholebodycontrol.md) | NVIDIA 官方训练/部署单仓（G1） |
| [Any2Any](../../wiki/entities/paper-any2any-cross-embodiment-wbt.md) | 同骨干、不同对齐/冻结合同的跨具身对照 |
