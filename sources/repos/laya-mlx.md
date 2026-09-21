# Laya-MLX（mizorewww/laya-mlx · Apple Silicon MLX 推理端口）

- **标题:** Laya-MLX
- **链接:** https://github.com/mizorewww/laya-mlx
- **类型:** repo / decision-engine / system-one / mlx-runtime
- **作者:** mizorewww（社区独立端口；上游 Laya 为 Convai Innovations / Nandha Kishor M）
- **许可:** Apache-2.0
- **PyPI:** https://pypi.org/project/laya-mlx/（`pip install laya-mlx`）
- **Hugging Face 权重（预转换 FP16 MLX）:**
  - https://huggingface.co/aac6fef/laya-mlx
  - https://huggingface.co/aac6fef/laya-multilingual-mlx
  - https://huggingface.co/aac6fef/laya-typed-decisions-mlx
- **上游 Laya:** https://github.com/NandhaKishorM/laya（pin commit `6a5819129eb220570792e417e49723d697efd76f`）
- **最后核查:** 2026-09-21
- **入库日期:** 2026-09-21

## 开源状态（步骤 2.5）

- **已开源：** GitHub 完整 MLX 推理/转换代码 + Apache 2.0 + `pip install laya-mlx` + HF 三 checkpoint 预转换权重；**非 Convai 官方发布**，为社区独立端口。
- **平台：** Apple Silicon（macOS 14+，Python 3.11+）；无 PyTorch / Transformers 运行时依赖。
- **训练：** 本仓仅推理与权重转换；RLCD 训练与微调仍在上游 [NandhaKishorM/laya](https://github.com/NandhaKishorM/laya)。

## 核心内容摘要

1. **MLX 原生 typed decision 推理：** 对 [Laya](../../wiki/entities/laya.md) 三 checkpoint 做独立 MLX 重实现；单次前向输出 `choice` / `score` / `noul`，**0 output tokens**。
2. **M3 Max 延迟（FP16，含 tokenize + 同步推理 + 校准）：** 英文 421M P50 **13.42 ms**；多语言 322M P50 **7.39 ms**；50 问吞吐 146.8 / 395.0 q/s（`batch_size=64`）。
3. **数值保真：** 63/63 验证题 FP32/FP16 与上游选中标签一致（378/378）；100 次重复调用零 active-memory 增长（fixture 级，非全空间保证）。
4. **HF 预转换权重：** `aac6fef/laya-mlx` 等三仓可直接 `laya.load("aac6fef/laya-mlx")`；亦支持 `convaiinnovations/laya` 原 ID + `laya-mlx convert`。
5. **Snake 演示：** `laya-snake` 终端 demo；`--optimize --max-speed` 在 M3 Max 上 **75.40 moves/s**（2400 步、零死亡）。
6. **Router / CLI：** 自上游适配 `Router`、语言启发式、`laya-mlx predict` CLI；`compile` / `cache_prompts` / `pad_to_multiple` 可选优化路径。

## 对 wiki 的映射

- **wiki/entities/laya-mlx.md** — MLX 运行时实体（与 [laya](../../wiki/entities/laya.md) 对照）
- **wiki/entities/laya.md** — 补交叉引用：Apple Silicon 本地部署选型
- **wiki/concepts/llm-robotics-control-interfaces.md** — 毫秒级本地 System 1 门控
