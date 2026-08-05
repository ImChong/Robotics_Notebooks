# BotRunner64/Teleopit（全身人形遥操作主框架）

- **标题**: Teleopit — full-embodiment humanoid teleoperation
- **类型**: repo / humanoid / teleoperation / motion-tracking / mjlab / unitree-g1
- **作者**: BotRunner64（西湖大学 / 上海创智学院团队维护）
- **机构**: 西湖大学、上海创智学院（见 [Teleopit 论文](../papers/teleopit_arxiv_2608_01834.md)）
- **链接**: <https://github.com/BotRunner64/Teleopit>
- **项目页**: <https://botrunner64.github.io/teleopit-page/>
- **文档**: <https://BotRunner64.github.io/Teleopit/>（[中文](https://BotRunner64.github.io/Teleopit/zh-Hans/)）
- **许可证**: Apache-2.0
- **默认分支**: `master`
- **收录日期**: 2026-08-05

## 一句话摘要

面向 **Unitree G1** 的轻量可扩展 **全身遥操作** 框架：BVH / Pico 4 VR → 实时运动重定向，支持 **MuJoCo sim2sim** 与真机 **sim2real**；配套 OpenNeck、LinkerHand O6、pico-bridge 与高层策略宿主运行时。

## 为何值得保留

- **论文可复现主入口**：与 arXiv:2608.01834 / 项目页五仓栈对齐；README 提供最小 sim2sim 与资产下载脚本。
- **发布权重**：`ckpt/track_g1.{pt,onnx}`（默认 29 DoF）与 `ckpt/track_g1_neck_o6.{pt,onnx}`（颈+O6 变体）。
- **采数闭环**：manifest 式录制（`schema.json` / `episodes.jsonl` / HDF5 / MP4），可接 [lerobot-teleopit](https://github.com/BotRunner64/lerobot-teleopit)。
- **生态引用**：被 [OASIS](../../wiki/entities/paper-loco-manip-04-oasis.md) 用作仿真 VR teleop 低层 WBC；[MimicLite](../../wiki/entities/mimiclite.md) 列为跨 codebase 部署策略之一。

## 快速路径（README）

```bash
pip install -e .
pip install modelscope
python scripts/setup/download_assets.py --only robots gmr ckpt bvh
python scripts/run/run_sim.py \
    controller.policy_path=ckpt/track_g1.onnx \
    input.bvh_file=data/sample_bvh/aiming1_subject1.bvh
```

## 版本锚点（维护）

| 版本 | 日期 | 要点 |
|------|------|------|
| v0.5.0 | 2026-08-03 | 宿主高层策略 msgpack/ZMQ 运行时；OpenNeck 0.2.0；somehand 0.3.0；匹配 G1/neck-O6 checkpoint |
| v0.4.0 | 2026-06-25 | pico-bridge 0.2.1；LinkerHand L6/O6；rewind sampling 训练路径 |
| v0.1.0 | 2026-03-25 | 首发：General-Tracking-G1、ONNX sim2sim、Pico 4、真机部署 |

## 配套仓库

| 仓 | 角色 |
|----|------|
| [somehand](./somehand.md) | 灵巧手优化重定向 |
| [pico-bridge](https://github.com/BotRunner64/pico-bridge) | PICO↔PC 传感桥 |
| [OpenNeck](https://github.com/BotRunner64/OpenNeck) | 2-DoF 主动颈硬件与驱动 |
| [lerobot-teleopit](https://github.com/BotRunner64/lerobot-teleopit) | LeRobot 数据集转换与 ACT/GR00T 训练 |

## 对 Wiki 的映射

- 论文实体：[paper-teleopit.md](../../wiki/entities/paper-teleopit.md)
- 项目页：[teleopit-project.md](../sites/teleopit-project.md)
- 论文归档：[teleopit_arxiv_2608_01834.md](../papers/teleopit_arxiv_2608_01834.md)
