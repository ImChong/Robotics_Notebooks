# 具身智能入门③（下）：读懂 PPO 训练产出，导出 ONNX 交给边缘端

> 来源归档（blog / 微信公众号 · 智践行）

- **标题：** 具身智能入门③（下）：读懂 PPO 训练产出，导出 ONNX 交给边缘端
- **类型：** blog
- **作者：** 智践行
- **原始链接：** https://mp.weixin.qq.com/s?__biz=Mzk2NDU0MzA3OA==&mid=2247492496&idx=1&sn=767310ed64bd0345a7dc42381f9fe597
- **专辑：** [wechat_zhixing_microduck_primer_album](../raw/wechat_zhixing_microduck_primer_album_4688586645438726146.md)
- **入库日期：** 2026-10-02
- **抓取方式：** 正文 URL 2026-10-02 CAPTCHA；④ 引用本篇 export 命令与 `infer_policy.py` 体验；日志读法对齐 [microduck_rl AGENTS.md](https://github.com/pollen-robotics/microduck_rl/blob/develop/AGENTS.md) § Training ops
- **一句话说明：** 用 wandb 读懂 PPO 惩罚项符号与课程节奏，经 **唯一安全路径** `scripts/export.py` 得到 `[1,61]→[1,14]` ONNX，并在 CPU 仿真里无头验证推理链。

## 核心摘录（归纳）

### 导出（红线）

```bash
uv run scripts/export.py Mjlab-Velocity-Flat-MicroDuck --checkpoint 199
# 或 --wandb-run-path <entity/project/run_id>
```

- **禁止**手写 `torch.onnx.export`：观测归一化必须烤进图；`play` 会掩盖未归一化 checkpoint。

### 日志怎么读（AGENTS 浓缩）

| 信号 | 含义 |
|------|------|
| 每个 `Episode_Reward/*` **≤ 0** | 惩罚项权重符号正确（违反则策略会「刷分」） |
| 主任务项随迭代上升 | 总 reward 升但 trick 未发生 = 被正则项骗 |
| 课程边界 metric 下跌 | 阶段 pacing 过早，应拉长 stage |

### 仿真侧闭合

```bash
xvfb-run -a uv run scripts/infer_policy.py --walking output.onnx --new-cmd-obs
```

- 与 ④ 中 Rust `policy-rehearsal` 对照：同一 ONNX、同一 61D 观测，应用 `--save-csv` 取轨迹做跨端对齐。

## 对 wiki 的映射

- 详情页：[zhixing-microduck-primer-part3b-ppo-logs-onnx-export.md](../../wiki/overview/zhixing-microduck-primer-part3b-ppo-logs-onnx-export.md)
- 概念：[reward-design.md](../../wiki/concepts/reward-design.md)、[sim2real.md](../../wiki/concepts/sim2real.md)
