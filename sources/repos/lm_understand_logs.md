# lm_understand-logs 实验资料仓库

- **地址：** <https://github.com/RikiyaT/lm_understand-logs>
- **项目原文：** <https://rikiyat.github.io/blog/weight-space-programming/>
- **核查日期：** 2026-11-15
- **内容：** 8 组实验日志、提示、模型输出、部分权重、评测与脚本。仅报告的实验运行在库内，早期提示词试验不在其中。
- **重建入口：** `python exp8_math/code/apply_edit.py <base model.safetensors> <run edit dir> <out>`；读取 manifest、diff 与基座张量，验证输入和重建结果的 SHA-256，再输出 safetensors 或 npy。
- **Gemma 对照：** [exp8_math/gemma_followup/README.md](https://github.com/RikiyaT/lm_understand-logs/blob/main/exp8_math/gemma_followup/README.md) 给出基座 `google/gemma-3-270m-it`、版本 `ac82b4e820549b854eebf28ce6dedaf9fdfa17b3`、运行 `g05b/g10b/g25b/g50b`、独立评测及 `eval/*/merged.json`；MATH-500 6.8% → 7.0% / 7.6% / 7.7% / 7.0%，未确立显著提升。可用同仓 `exp8_math/code/apply_edit.py` 校验差分重建。
- **HF 检验：** [发布模型卡](https://huggingface.co/RikiyaT/Qwen3.5-0.8B-weight-space-programming) 记载 Qwen 编辑权重、两处变动张量的哈希及 H200 完整 MATH-500 的重测；[在线 Demo](https://huggingface.co/spaces/akhaliq/weight-space-programming-demo) 展示基础/编辑版对比（并非统计评测）。
- **评测：** `exp8_math/code/evaluate.py` 列出 prepare/run 入口；README 说明 think35、vLLM、数据和评分。该脚本引用未包含的 `weightenv` 模块，仓库亦声明 agent harness 不含，不能假定全流程开箱即用。
- **许可：** 仓库 README 标注自有脚本 MIT，日志、数据和网络 CC BY 4.0；第三方内容另有许可。
- **关联：** [博客](../blogs/weight_space_programming.md)；[项目页](../sites/weight-space-programming.md)；[实体](../../wiki/entities/weight-space-programming.md)。
