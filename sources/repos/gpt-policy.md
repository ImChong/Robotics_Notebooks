# GPT-Policy（cheng-haha/GPT-Policy）

> 来源归档

- **标题：** GPT-Policy
- **类型：** repo
- **链接：** <https://github.com/cheng-haha/GPT-Policy>
- **论文：** <https://arxiv.org/abs/2609.19138>
- **项目页：** <https://cheng-haha.github.io/GPT-Policy/>
- **入库日期：** 2026-09-18
- **一句话说明：** 固定 VLM 的 in-context 机器人代理：context compiler、tool 协议、ARX/YAM Cartesian adapter、`gpt-policy` CLI。
- **沉淀到 wiki：** [`wiki/entities/paper-gpt-policy.md`](../../wiki/entities/paper-gpt-policy.md)

---

## 仓库入口（README）

| 组件 | 说明 |
|------|------|
| 安装 | `pip install -e .`；可选 `scripts/install_drivers.py arx|yam|realsense` |
| 默认运行 | `gpt-policy "pick up the red block"`（读 `configs/default.json`） |
| 配置检查 | `gpt-policy --check` |
| JSON 任务包 | `gpt-policy --input-json task.json` |
| 源码 | `src/gpt_policy/` — protocol、planning、adapters、recording |
| 支持平台 | ARX X5、I2RT/YAM |

## 开源边界（截至 2026-09-18）

- **已开源：** 完整 harness、格式 adapter、真机 ARX 管线。
- **外部依赖：** 商业 VLM API（论文主实验 GPT-6 Astra）；credentials 不入库。
- **评测规模：** 项目页十任务 × 三 trials/condition；非大规模 benchmark。
