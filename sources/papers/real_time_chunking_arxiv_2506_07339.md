# Real-Time Execution of Action Chunking Flow Policies（arXiv:2506.07339）

> 来源归档（ingest）

- **标题：** Real-Time Execution of Action Chunking Flow Policies
- **短名：** Real-Time Chunking / RTC
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2506.07339>
- **项目页：** <https://www.pi.website/research/real_time_chunking>
- **代码（仿真）：** <https://github.com/Physical-Intelligence/real-time-chunking-kinetix>
- **训练期后续：** <https://arxiv.org/abs/2512.05964>（博客 2025-12-08 更新；用于 π*₀.₆ 咖啡演示）
- **机构：** 物理智能（Physical Intelligence）；加州大学伯克利分校（UC Berkeley）
- **入库日期：** 2026-09-28
- **一句话说明：** 推理期把下一段动作 chunk 当成补全问题，让 flow / diffusion VLA 在执行旧 chunk 时生成与之衔接的新 chunk。

## 开源状态（步骤 2.5，2026-09-28）

- **部分开源**：Kinetix 仿真实验仓可跑专家数据、flow 模仿与延迟扫描，并含 training-time RTC 的微调步骤。权重与数据在 `gs://rtc-assets/`（`expert/` 约 60GiB）。
- 真机 π₀ / π₀.₅ 上的 inpainting 推理**未**作为 openpi 的独立训练入口发布。博客写明 π₀、π₀-FAST、π₀.₅ 的定量结果当时仍是同步执行。

## 核心摘录（面向 wiki 编译）

- π 系 chunk 为 50 步、约 1 秒。同步执行会在 chunk 之间停顿；停顿不在训练分布里。时间集成（ACT 的 temporal ensembling）在高延迟下会给出无效动作。
- RTC 在推理时冻结已经来不及改的前几步（对齐上一 chunk），对其余重叠步部分约束，再补全剩余动作。不改训练即可套到 flow / diffusion VLA。
- 博客示例延迟：移动操作远程推理合计约 139 ms（模型 97 ms），固定臂约 108 ms。注入 +200 ms 后 RTC 吞吐基本不变，同步推理与 temporal ensembling 明显下降。
- **对 wiki 的映射：** [paper-real-time-chunking](../../wiki/entities/paper-real-time-chunking.md)；仓库归档 [real-time-chunking-kinetix](../repos/real-time-chunking-kinetix.md)
