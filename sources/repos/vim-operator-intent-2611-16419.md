# I See What You Want: Decoding Hierarchical Operator Intent from Visual Teleoperation History 官方源码核对

- **类型：** 官方源码仓
- **代码：** https://github.com/see-want/VIM
- **论文：** https://arxiv.org/abs/2611.16419
- **公开运行入口：** `vim-infer --base-model checkpoints/qwen3.5-27b --model checkpoints/vim27b_lora_v2`
- **相关模块：** `src/vim 的推断与序列编码`
- **运行产物：** ongoing action / mission
- **核对范围：** 官方 README、该仓库公开入口脚本与关键模块的静态核查；外部数据和模型依赖未执行。
- **通读：** 已完成
- **wiki：** [论文独立详情](../../wiki/entities/paper-vim-operator-intent-2611-16419.md)
- **原文：** [论文来源](../papers/vim-operator-intent-2611-16419.md)

官方 VI AMO 数据需获得访问权限，模型训练需较大显存；不能把人工确认加微调的接管称为自主零样本接管。
