# LongCat-Next 官方实现与可运行入口

- **仓库：** <https://github.com/meituan-longcat/LongCat-Next>
- **模型权重：** <https://huggingface.co/meituan-longcat/LongCat-Next>
- **推理实现：** <https://github.com/meituan-longcat/LongCat-Next-inference>
- **训练入口：** <https://github.com/meituan-longcat/LongCat-Next/blob/main/sft/train.py>
- **训练说明：** <https://github.com/meituan-longcat/LongCat-Next/blob/main/sft/README.md>
- **类型：** 官方模型代码与 SFT 工程
- **入库日期：** 2026-11-10
- **核对范围：** 仓库 README 的 model introduction、Quick Start、文本/图像/音频生成样例及 SFT README 的三模式训练、分布式入口和数据结构；未逐行审计全部底层算子。
- **通读：** 已完成

## 模型推理入口（README）

`transformers.AutoModelForCausalLM.from_pretrained("meituan-longcat/LongCat-Next", trust_remote_code=True)` 加载模型；`AutoTokenizer` 应用聊天模板；`AutoProcessor` 将文本、图像、音频转换为输入；`model.generate(input_ids, visual_inputs, audio_inputs)` 生成；按文本、`visual_ids` 或 `audio_ids` 分别解码。README 提醒权重按需加载，模型示例建议至少 3×80GB GPU。

## SFT 训练入口

`torchrun --nproc_per_node=8 --nnodes=1 --master_port=29517 train.py --task unify --model_path ... --data_path ...`

- `sft/train.py` 配 `sft/config.py` 与数据集/packing 模块，进行 FSDP2 分片和模型加载。
- `understand`：图像+文本→文本（交叉熵）。
- `generate`：文本→图像 token（depth cross-entropy）。
- `unify`：混合样本在同一次训练里按样本计算损失并聚合。
- `sft/examples/train_unify_8gpu.sh` 为具体样例；README 报告 8×H800 80GB 测试环境。
- 许可证：GitHub README 说明 MIT；使用权重前仍应核对 HF 上的许可证与附加条件。

## 边界

模型和训练**存在可运行入口**，但本次未在 GPU 上执行或复现实验。SFT 开源不代表完整的多阶段预训练流水线和训练语料全部公开。

**对应 wiki：** [LongCat-Next](../../wiki/entities/paper-longcat-next.md)
