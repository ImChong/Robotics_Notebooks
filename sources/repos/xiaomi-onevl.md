# Xiaomi OneVL：官方推理与训练代码

- **类型：** repo
- **推理仓库：** <https://github.com/xiaomi-research/onevl>
- **训练仓库：** <https://github.com/GeorgeLuImmortal/OneVL_training>（由官方推理仓 README 链接）
- **论文：** <https://arxiv.org/abs/2604.18486>
- **项目：** <https://xiaomi-embodied-intelligence.github.io/OneVL/>
- **模型权重：** <https://huggingface.co/collections/xiaomi-research/onevl-models>
- **代码许可：** 官方仓库 README 标示 Apache-2.0；模型、数据、Emu3.5 tokenizer 另遵守各自许可
- **入库日期：** 2026-11-09

## 源码核对范围与可复现入口

- 推理仓库 README：Python/CUDA 环境、权重、单卡/多卡命令、分数据集脚本、解释开关、NAVSIM 转换程序。
- `infer_onevl.py`：`main()`、`load_test_set()`、`compute_inference_latent_positions()`、`decode_latent_with_aux()`、`decode_latent_with_visual_aux()`、`FloatMLPHead`；模型用 `Qwen3VLForConditionalGeneration.from_pretrained` 加载，`AutoProcessor` 构造图像与对话输入，`model.generate` 产生轨迹 token。
- 代码将 `--num_latent_vis=4` 与 `--num_latent=2` 转成现有词表的潜标记字符串；默认主规划无需辅助解释模型。`--decoder_explain` / `--visual_decoder_explain` 打开额外的前向取隐藏态与对应辅助解码；`--float_head_type` 另有 MLP waypoint 分支。
- `run_infer.sh`：按数据分片→逐 GPU 启动 `infer_onevl.py`→合并预测 JSON。
- `eval_results.py` 评估 ROADWork、Impromptu、APR1；NAVSIM 需 `output/navsim/convert_to_eval.py` 转换输出，再接官方 [NAVSIM](https://github.com/autonomousvision/navsim) 评测流水线。
- 训练仓库 README 及 `run_script/train/navsim/sft_distributed_stage2_vis4_txt2_bs64.sh`：Stage 0 基座 warm-up，Stage 1 主干冻结训练辅助头，Stage 2 `swift sft --model_type qwen3_vl_latent_cot` 联合微调，配置视觉/语言潜标记数量 4/2、对应损失权重 0.1/1.0。

## 工程注意

主规划速度指标不包括可选的语言/视觉解码开销。解释模式额外需要辅助权重和一次 `output_hidden_states=True` 的前向，因此不能将解释模式延迟直接等同于论文报告的 4.46 秒。README 给出辅助解码器推理建议至少 16 GB 显存；未在本次入库中运行 GPU benchmark、训练或 NAVSIM 全量复现。

## 对 wiki 的映射

- [OneVL 实体页（含源码时序图）](../../wiki/entities/paper-xiaomi-onevl.md)
- [论文原文](../papers/onevl_arxiv_2604_18486.md)
- [官方项目页](../sites/xiaomi-onevl.md)
