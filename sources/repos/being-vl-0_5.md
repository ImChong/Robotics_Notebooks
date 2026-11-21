# Being-VL-0.5 仓库核查

> 来源归档（repo / BeingBeyond GitHub）

- **URL：** <https://github.com/BeingBeyond/Being-VL-0.5>
- **机构 / 个人：** BeingBeyond（智在无界）
- **仓库核查commit：** `e92ce36`（1,817 个文件，包含 1,740 个 `.py`/`.sh` 源码与脚本）
- **开源结论：** **已开源完整训练与评测代码及 Hugging Face 模型权重**（`BeingBeyond/Being-VL-0.5-8K`、`16K`）。
- **核对范围：** 已通过 `gh api` 核对 `BeingBeyond/Being-VL-0.5 @ e92ce36` 文件树、根目录 `README.md` 及词表转换/训练/评测入口命名；未实际运行 GPU 训练或评测脚本。
- **通读：** 已完成

## 核心源码入口与模块

- **词表与模型转换**：`convert_llama_beingvl.py`、`train_vbpe.py`（Priority-guided vBPE 构建）、`tokenize_dataset.py`（VQ-GAN 离散编码序列化）
- **三阶段课程训练**：`train-stage-1.sh`（嵌入对齐）、`train-stage-2.sh`（浅层 25% 解冻混合）、`train-stage-3.sh`（全参多模态指令微调）
- **推理与基准评测**：`eval_vqa.sh` 及 `llava/` 多模态训练与评测管线
