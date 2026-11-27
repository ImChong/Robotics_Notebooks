# LAMR 作者源码核对

- **链接：** <https://github.com/PengyuHuang1/LAMR>
- **固定提交：** [4b7bd5b](https://github.com/PengyuHuang1/LAMR/tree/4b7bd5b1cd4162d66e5f16c8560781fe1139e026)
- **许可证：** BSD-3-Clause（仓库许可证）
- **论文：** <https://arxiv.org/abs/2611.11907v1> · [全文归档](../papers/lamr_arxiv_2611_11907.md)
- **项目页：** <https://pengyuhuang1.github.io/LAMR/> · [项目页归档](../sites/lamr.md)
- **核查日期：** 2026-11-27

## 发布的边界

- `lamr_layer_role_profile.py` 依赖已有 AffectGPT 环境、检查点、编码器与 CMU 数据。它采集注意力、按模态 token 分组计算 ME/CMI，聚合每样本的层排序并输出含 `integration_layers` 的 JSON。README 示例为 CMU-MOSI test 均匀取 300 项，与论文 §4.1.2 的 MOSI+MOSEI 校准集不同。CLI 默认 top-13；使用 `--use-kneedle` 才进入内置拐点分支。
- `modeling_qwen2.py` 是给 AffectGPT 的简化 eager 补丁：token 元数据与长度条件满足时走重配，在因果掩码和 softmax **之前**处理 logits；整合层清单为内置 11 个索引，两个分支均用 0.2。未发现自动读入上述画像 JSON，不能把公开示例视为论文设置的逐项复刻。整合分支的局部键掩码结果随后被完整扩散矩阵覆盖。
- README 的 TODO 与文件内容不同步：虽列「画像代码未发布」，固定提交确有独立脚本；整套修改后的推理、其他骨干入口和完整评测未见发布于此仓库。需要上游 AffectGPT 模型和依赖。

## 本次验证

从固定提交运行原样提取的 `select_integration_layers` 纯 NumPy 函数：验证 top-k、内置拐点以及并列分数在两分支的选择行为，共 4 项断言通过；两个 Python 文件均通过语法编译。未安装模型环境、未载入权重，也未复跑论文数据或 Qwen2 注意力算子。

- **沉淀到 wiki：** [LAMR](../../wiki/entities/paper-lamr.md)
