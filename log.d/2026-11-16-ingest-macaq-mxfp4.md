## [2026-11-16] ingest | MACaQ | 逐层 MXFP4 量化的精度、时间与推理代价

- 意图：依据 arXiv:2611.10649v2 完整方法与实验收录 MACaQ，并核实多后端官方实现。
- 开源结论：论文、vLLM/llama.cpp/Diffusers/PyTorch 代码与部分 HF 模型产物可访问；未实际复现。
- 知识定位：新增 [MACaQ 独立论文详情](../wiki/entities/paper-macaq.md)，挂载模型架构纵深路线 Stage 5；与机器人 VLAQuantBench 闭环量化评测对照，避免把语言模型 PPL 误写成真机成功率。
- 一手资料：https://arxiv.org/abs/2611.10649v2；https://github.com/huawei-csl/MACaQ；https://huawei-csl.com/blog/macaq/；https://huggingface.co/collections/huawei-csl/macaq
