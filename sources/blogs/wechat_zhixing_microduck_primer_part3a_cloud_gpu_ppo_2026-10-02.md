# 具身智能入门③（上）：不买显卡也能练，免费云 GPU 跑通 Microduck 的 PPO 训练

> 来源归档（blog / 微信公众号 · 智践行）

- **标题：** 具身智能入门③（上）：不买显卡也能练，免费云 GPU 跑通 Microduck 的 PPO 训练
- **类型：** blog
- **作者：** 智践行
- **原始链接：** https://mp.weixin.qq.com/s?__biz=Mzk2NDU0MzA3OA==&mid=2247492482&idx=1&sn=f2f318a943b8f8ae45a48a6907cabda6
- **专辑：** [wechat_zhixing_microduck_primer_album](../raw/wechat_zhixing_microduck_primer_album_4688586645438726146.md)
- **入库日期：** 2026-10-02
- **抓取方式：** 正文 URL 2026-10-02 CAPTCHA；④ 文内引用魔搭云实例 `/mnt/workspace/setup_env.sh`；命令对齐 [microduck_rl README](https://github.com/pollen-robotics/microduck_rl) 与 `AGENTS.md` smoke 约定
- **一句话说明：** 在云端 GPU（含国内魔搭免费实例）克隆 `microduck_rl`，先 64 env × 5 iter smoke，再跑主任务 `Mjlab-Velocity-Flat-MicroDuck` PPO 训练。

## 核心摘录（归纳）

### 环境前提

- **CUDA + uv**；MuJoCo Warp 训练走 GPU。ARM 盒首次 `uv sync` 需 `UV_HTTP_TIMEOUT=600`（见 `AGENTS.md`）。
- 无本地 GPU 时官方另提供 `train … --hf-jobs`（Hugging Face Jobs）。

### 推荐命令链

```bash
git clone https://github.com/pollen-robotics/microduck_rl && cd microduck_rl
uv sync
# smoke（必须先跑，抓配置/维度错误）
uv run train Mjlab-Velocity-Flat-MicroDuck --env.scene.num-envs 64 --agent.max_iterations 5
# 正式 walk（4096 envs，约 1–2 h 可用步态量级）
uv run train Mjlab-Velocity-Flat-MicroDuck --env.scene.num-envs 4096
```

### 云实例注意点（④ 交叉验证）

- 训练产物、checkpoint、后续 `infer_policy.py --save-csv` 在 **云盘**；导出 ONNX 后需下载到本机做 Rust 部署篇。
- 无头推理常配 `xvfb-run`；`infer_policy` 无头下 **Ctrl+C** 才落盘 CSV（按 Q 无效）。

## 对 wiki 的映射

- 详情页：[zhixing-microduck-primer-part3a-cloud-gpu-ppo-training.md](../../wiki/overview/zhixing-microduck-primer-part3a-cloud-gpu-ppo-training.md)
- 实体：[pollen-microduck-rl.md](../../wiki/entities/pollen-microduck-rl.md)、[mjlab.md](../../wiki/entities/mjlab.md)
