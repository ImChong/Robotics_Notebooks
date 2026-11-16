# MimicLite — RoboParty 官方使用指南

> 官方技术文档一手资料归档（非转载全文）；2026-11-11 核查。

- **标题：** MimicLite 使用指南 / MimicLite (RoboParty Party OS)
- **类型：** site（官方技术文档，安装、数据、训练、部署、评测全链路）
- **官方文档：** https://roboparty.com/documents/party-os?doc=mimiclite&lang=zh
- **官方项目仓库：** https://github.com/Roboparty/MimicLite
- **配套训练源码：** https://github.com/EGalahad/mimic-lite
- **配套部署源码：** https://github.com/EGalahad/sim2real
- **核对日期：** 2026-11-11
- **核对范围：** 完整通读 RoboParty 官方《MimicLite 使用指南》全文，第 1–6 章（概览、数据、PPO/PPO-ROA 训练、查看与 ONNX/YAML 导出、MuJoCo sim2sim / Unitree G1 部署和 PICO 遥操作、MotionDecode 评测）；交叉核对 Roboparty/MimicLite 英/中文 README、EGalahad/mimic-lite README、docs/checkpoints.md 与 EGalahad/sim2real README。核验仓库内训练、分阶段训练、查看导出和评测入口。
- **通读：** 已完成
- **开源状态：** 项目已开源；项目主仓库是 GPL-3.0-or-later 集成入口，训练、数据转换及推理分别在外部仓库，各有独立许可证。BONES-SEED 数据另有许可，不可随意再分发。
- **沉淀到 wiki：** [MimicLite](../../wiki/entities/mimiclite.md)
- **源码归档：** [mimiclite.md](../repos/mimiclite.md)

## 原文对应的技术路线

1. **Motion Data：** `any4hdmi` 的 `manifest.json` + `motions/*.npz`；数据包含 qpos 和机器人 MJCF 引用，训练端进行 FK 缓存；支持 HF `hf://`、PICO 录制和 BONES-SEED 本地转换。
2. **PPO：** `active-adaptation` 中 `projects/mimic-lite/scripts/train.py`，Hydra 机器人/动作/算法配置和多 GPU `launch_ddp.sh`。支持 mjlab 和 IsaacLab 两后端。
3. **PPO-ROA：** `train_sequential.py` 串联教师（可见特权信息/更多未来帧）→ Adapt 学生 → PPO Finetune；v1.1 阶段 4000/1000/2000 updates、每卡 8192 个环境、8 GPU。
4. **导出：** `scripts/play.py` 的 `export_policy=true` / `export_only=true`，同次导出生成策略 `.onnx` 和部署 `.yaml`，后者记录观测、关节顺序、PD 增益和动作尺度。
5. **部署：** `sim2real` 的 `sim_env/base_sim.py` + `rl_policy/tracking.py` 使 MuJoCo 与真实 Unitree G1 共享观测协议；PICO 经 `teleop/pico_retarget_pub.py` 提供参考运动。
6. **统一评测：** MotionDecode 的 Locomotion-80、Manipulation-48、Ground-60、Dance-40；MuJoCo 中运行导出的 ONNX，比较完整度和误差；失败帧之后不计误差，所以要同时看完成度。

## 结果口径与边界

- **3 h on 8×RTX 4090：** 官方主页/中文 README 的“约三小时训练”是项目总述，不能直接当作每份发布权重的测量时长。
- **实测不同发布版本：** 官方实施仓库 README 报告 G1 PPO 92.3 GPU-h、G1 PPO-ROA 173.2 GPU-h、G1 v1.1 50.106 GPU-h；三者模型规模和环境数不相同。
- **Future reference：** MimicLite PPO/ROA/v1.1 均只需要未来 0.08 s 的参考运动；PPO-ROA 的教师可在训练期看到更长未来（0.32 s），部署学生仍是 0.08 s。
- **G1 限制：** 官方对比图只覆盖 G1；其他机器人发布了策略但不等于匹配的跨机器人基准数据。
- **A3 特别限制：** 配套 checkpoint 文档称 A3 在单独 native 测试 67/72 通过，长时序存在 root drift，集成 ONNX benchmark 与 sim2real 部署未验证，不能外推 A3 真机效果。
- **未覆盖：** `mimic-lite.pdf` 24MB 技术报告因当前读取链路无法取得全文，本次没有通读 PDF、也未独立跑训练或实机。**上面的「通读：已完成」仅针对本条标明的官方技术文档和配套 README/源码入口，不代表读完 PDF。**

## 一手资料

- [官方完整使用指南](https://roboparty.com/documents/party-os?doc=mimiclite&lang=zh)
- [RoboParty MimicLite 入口 README](https://github.com/Roboparty/MimicLite/blob/main/README.md)
- [官方训练代码 README](https://github.com/EGalahad/mimic-lite/blob/main/README.md)
- [机器人与 checkpoint 明细](https://github.com/EGalahad/mimic-lite/blob/main/docs/checkpoints.md)
- [官方部署运行时 README](https://github.com/EGalahad/sim2real/blob/main/README.md)
