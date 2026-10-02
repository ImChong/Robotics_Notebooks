# 具身智能入门②：不写一行代码，先把机器鸭 Microduck「玩」明白

> 来源归档（blog / 微信公众号 · 智践行）

- **标题：** 具身智能入门②：不写一行代码，先把机器鸭 Microduck「玩」明白
- **类型：** blog
- **作者：** 智践行
- **原始链接：** https://mp.weixin.qq.com/s?__biz=Mzk2NDU0MzA3OA==&mid=2247492360&idx=1&sn=dae6b7ed7716bfb6fdb2c78279cca5f6
- **专辑：** [wechat_zhixing_microduck_primer_album](../raw/wechat_zhixing_microduck_primer_album_4688586645438726146.md)
- **入库日期：** 2026-10-02
- **抓取方式：** 正文 URL 2026-10-02 CAPTCHA；归纳对照 [microduck_rl README](https://github.com/pollen-robotics/microduck_rl/blob/develop/README.md) `play` / `infer_policy.py` 与 [microduck simulation.md](https://github.com/pollen-robotics/microduck/blob/main/docs/robot/simulation.md)
- **一句话说明：** 在 MuJoCo 里用键盘/官方 ONNX 彩排行走与技能切换，建立「61 维观测 → 14 路动作 → 50 Hz」数据流直觉，无需改代码、无需真机。

## 核心摘录（归纳）

### 零代码入口

| 命令 | 作用 |
|------|------|
| `uv run play Mjlab-Velocity-Flat-MicroDuck --wandb-run-path …` | GPU 仿真 + 查看器，看已训策略 |
| `uv run scripts/infer_policy.py --walking output.onnx` | **CPU MuJoCo** 键盘驱动，排练部署合同 |
| `scripts/duck-sim`（Runtime 仓） | 真机 daemon 对 MuJoCo 身体全栈仿真（进阶） |

### infer_policy 键盘语义（与真机一致）

- 速度 twist、`G` 触地拾取、`Y` 坐站、`R` 翻滚、`K`/`L` 踢球；多策略 `--walking/--standing/...` **热切换**。
- `--new-cmd-obs` 与 Runtime 命令槽写入方式对齐；全零 twist 在部署侧常对应 **站立 idle**（勿误以为「策略坏了」）。

### 玩明白什么

- **观测 61D / 动作 14D** 是全家桶硬合同（热切换前提）。
- BAM 电压舵机仿真默认开；`--no-bam` 才回 XML PD，与训练不一致时 sim2real 会骗你。

## 对 wiki 的映射

- 详情页：[zhixing-microduck-primer-part2-play-without-code.md](../../wiki/overview/zhixing-microduck-primer-part2-play-without-code.md)
- 实体：[pollen-microduck-rl.md](../../wiki/entities/pollen-microduck-rl.md) § infer / 61D 合同
