# Cartwheel Control：官方 Hugging Face 模型卡和 README

> 一手资料归档（2026-11-12）

- **标题：** Cartwheel Control
- **类型：** site / model card
- **机构：** Cartwheel
- **模型与权重：** <https://huggingface.co/andrew-cartwheel/cartwheel-control>
- **官方 README：** <https://github.com/Cartwhl/cartwheel-control/blob/main/README.md>
- **代码：** <https://github.com/Cartwhl/cartwheel-control>
- **介绍博客：** <https://getcartwheel.com/blog/cartwheel-control>（未能获取完整正文）
- **核对范围：** 已通读 Hugging Face 官方模型卡和 GitHub README 全部公开正文，核对控制接口、型号、资产文件、运行步骤、动作格式、许可和仿真边界；未读博客全文。
- **通读：** 已完成
- **入库日期：** 2026-11-12
- **一句话说明：** Cartwheel Control 提供三种型号的 50 Hz 全身运动跟踪策略，在物理仿真里执行参考动作。

## 原文要点

1. **控制接口：** 历史关节与 IMU 观测、未来 10 帧参考动作输入；每 20 ms 输出一组关节位置目标。
2. **已发布：** `rp1/`（28 DoF）、`asimov-1/`（25 DoF）、`g1/`（29 DoF），各目录有权重、配置、机器人模型、GLB 与示例 motion。
3. **运行：** `uv sync` 然后 `uv run cartwheel-play --robot rp1`；并行仿真、打印完成/跌倒状态、写入可视化网页。
4. **动作：** 50 Hz `.npz` 的 `qpos` 为 `(frames, 7 + joints)`，依次为根位置、wxyz 姿态和目标 MJCF 顺序的关节角。
5. **证据边界：** 模型卡明确仅仿真使用，尚未在实物机器人上运行；没有统一三款机器人成功率榜单。
6. **许可：** 代码 Apache-2.0；权重为 Cartwheel Open Model License，机器人模型另遵第三方条款。

## 开放边界

- **已开放：** 三份权重、示例轨迹、推理/仿真/可视化代码。
- **未开放：** 在此仓库中未发现可运行的训练程序，也未见完整训练数据和奖励/超参数定义。
- **未完成：** 官方介绍博客正文无法读取，不以其未核实的细节补全训练叙述。

## 对 wiki 的映射

- [Cartwheel Control 实体](../../wiki/entities/cartwheel-control.md)
- [官方源码归档](../repos/cartwheel-control.md)
