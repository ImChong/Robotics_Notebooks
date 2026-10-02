# 具身智能入门①：0硬件起步，从开源机器鸭 Microduck 学起

> 来源归档（blog / 微信公众号 · 智践行）

- **标题：** 具身智能入门①：0硬件起步，从开源机器鸭 Microduck 学起
- **类型：** blog
- **作者：** 智践行
- **原始链接：** https://mp.weixin.qq.com/s?__biz=Mzk2NDU0MzA3OA==&mid=2247492291&idx=1&sn=853662478d99f4390605c3ec55078d59
- **专辑：** [wechat_zhixing_microduck_primer_album](../raw/wechat_zhixing_microduck_primer_album_4688586645438726146.md)
- **入库日期：** 2026-10-02
- **抓取方式：** 正文 URL 2026-10-02 CAPTCHA；归纳对照 [microduck](https://github.com/pollen-robotics/microduck) / [microduck_rl](https://github.com/pollen-robotics/microduck_rl) README 与 [pollen-robotics-microduck.md](../sites/pollen-robotics-microduck.md)
- **一句话说明：** 不买整机也能建立 Microduck 心智模型：训练仓（Python/mjlab/PPO）与机载 Runtime（Rust/ONNX/50 Hz）两仓分工、61 维观测合同与 sim2real 主线。

## 核心摘录（归纳）

### 为什么要从 Microduck 入门

| 维度 | 读法 |
|------|------|
| 形态 | ~25 cm / ~800 g 桌面双足，14 路 XL330 进入 RL 图（整机 15 路舵机口径见 Runtime README） |
| 软件 | **已开源** Apache-2.0：`microduck_rl` 训策略、`microduck` 上板跑 ONNX |
| 商品 | 产品页卖整机；学习路径可 **零硬件** 从仿真与文档开始 |

### 两仓一张图

| 仓 | 语言 | 职责 |
|----|------|------|
| [microduck_rl](https://github.com/pollen-robotics/microduck_rl) | Python | MuJoCo Warp + PPO → `scripts/export.py` → ONNX |
| [microduck](https://github.com/pollen-robotics/microduck) | Rust | `robotd` 50 Hz 控制环、更新回滚、BLE/手柄 intent |

### 系列阅读顺序（本专辑）

① 认知栈 → ② 仿真里「玩」→ ③ 云 GPU 训 PPO → ③下 日志与 ONNX → ④ Rust mock 推理与 Python 对齐。

## 对 wiki 的映射

- 详情页：[zhixing-microduck-primer-part1-zero-hardware-start.md](../../wiki/overview/zhixing-microduck-primer-part1-zero-hardware-start.md)
- 专辑地图：[zhixing-microduck-primer-album-technology-map.md](../../wiki/overview/zhixing-microduck-primer-album-technology-map.md)
- 实体：[pollen-microduck.md](../../wiki/entities/pollen-microduck.md)、[pollen-microduck-rl.md](../../wiki/entities/pollen-microduck-rl.md)
