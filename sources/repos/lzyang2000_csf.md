# lzyang2000/csf 官方代码归档

> 来源归档（repo / 官方代码）

- **仓库：** <https://github.com/lzyang2000/csf>
- **项目页：** <https://lzyang2000.github.io/csf/>
- **论文：** <https://arxiv.org/abs/2610.12467>
- **许可：** 根仓库 Apache-2.0；third_party/ 中的生成器与模型权重遵循各自许可证。
- **核查日期：** 2026-10-12
- **一句话说明：** CSF 动作安全过滤器的公开实现，包含四类预训练动作生成器适配与浏览器交互 demo。

## README 运行路径

```bash
git clone --recurse-submodules https://github.com/lzyang2000/csf.git
cd csf
bash scripts/setup.sh
hf auth login
scripts/serve.sh
```

- 环境：Linux、NVIDIA GPU（24 GB）、uv；使用文本编码器前需接受 Hugging Face 上的 Llama 3 许可。
- Kimodo 与 ARDY 权重首次运行时下载；ECHO、MotionHiFlow 可选，可分别运行 scripts/fetch_echo_ckpts.sh、scripts/fetch_motionhiflow_ckpts.sh。
- 启动后在本地浏览器打开 http://localhost:7860；可选择示例或输入提示、勾选 Person in scene 检查门控，也可在播放动作时改变场景以演示运行时安全盾。
- 仓库包含 configs/rules.yaml、scripts/、csf/、tests/、third_party/；README quick start 指向网页 demo。论文另报告 ARDY + CSF + SONIC 的 G1 部署，硬件复现路径与 demo 环境须区分。

## 对 wiki 的映射

- [wiki/entities/paper-csf-contextual-safety-filter.md](../../wiki/entities/paper-csf-contextual-safety-filter.md) — 方法、实验、真机结果与复现边界
- [wiki/concepts/safety-filter.md](../../wiki/concepts/safety-filter.md) — 与控制输入域安全过滤的关系
