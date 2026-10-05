# LingBot-VA 官方代码与资源

- **类型：** repo
- **机构：** 蚂蚁灵波 Robbyant
- **核查日期：** 2026-10-05
- **项目页：** <https://technology.robbyant.com/lingbot-va/>
- **代码：** <https://github.com/robbyant/lingbot-va>
- **论文：** <https://arxiv.org/abs/2601.21998>
- **权重：** <https://huggingface.co/robbyant/lingbot-va-base>；另有 `lingbot-va-posttrain-robotwin`、`lingbot-va-posttrain-libero-long`。
- **后训练数据：** HF `robbyant/robotwin-clean-and-aug-lerobot`、`robbyant/libero-long-lerobot`；不是完整预训练池。
- **许可：** Apache-2.0 代码；数据与权重另核各资源许可。

## README 核查摘录

- 2026-01-29 发布 shared backbone；2026-02-17 发布后训练代码与数据；2026-04-08 / 04-24 补齐 LIBERO 路径与权重。
- 论文介绍 MoT 双流、因果交错序列、异步执行与缓存；已公开资源表明确写 shared backbone，不能推定分离双流版本全部开放。
- `script/run_va_posttrain.sh` 以 `CONFIG_NAME=robotwin_train` / `libero_train` 启动训练。
- `wan_va/configs/va_libero_cfg.py` 的 `action_snr_shift`、`used_action_channel_ids`、`norm_stat` 必须与最新权重对齐。
- README 的独立 `python inference.py` 示例已注释；当前仿真复现优先使用 README 各 benchmark 的 server-client 路径。

## 对 wiki 的映射

- [LingBot-VA](../../wiki/entities/paper-sa-2601-21998-lingbot-va-causal-video-action-world-model-for-g.md)
- [家族与 VA 2.0](../../wiki/entities/robbyant.md)
