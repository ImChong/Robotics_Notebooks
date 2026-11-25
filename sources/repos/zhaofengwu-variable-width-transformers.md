# Variable-Width Transformers 官方实现｜Official Reference Implementation

- **类型：** repo
- **代码：** <https://github.com/ZhaofengWu/variable-width-transformers>
- **版本：** `69dde8143d9a7912de353b57093d36d8788070d4`，`lm-engine` gitlink `64b843817bb9edaa30b9ffbf3f0686806904696c`
- **论文：** <https://arxiv.org/abs/2606.18246v1>
- **论文归档：** [arXiv v1](../papers/variable-width-transformers.md)
- **入库日期：** 2026-11-25
- **核对范围：** 官方 README、依赖、gitlink、稠密配置与 MoE 配置、`pretrain.py` / `pretrain.sh`、`custom_yaml_utils.py`、宽度求解和模型定义；静态阅读，未执行 GPU 训练。
- **通读：** 已完成

## 运行入口

- [README](https://github.com/ZhaofengWu/variable-width-transformers/blob/69dde8143d9a7912de353b57093d36d8788070d4/README.md)：模型、配置和预训练代码可获取；未见作者发布训练好的权重或独立许可证文件，代码可见不等于已授予任意再分发权限。
- [训练入口](https://github.com/ZhaofengWu/variable-width-transformers/blob/69dde8143d9a7912de353b57093d36d8788070d4/pretrain.py)：`python pretrain.py --config configs/dense_500m.yml`；多卡包装脚本以 `torchrun -m pretrain --config` 启动同一路径。YAML 先展开全局变量到独立文件；续训目录无迭代标记时从生成配置移除加载选项，然后交给 lm-engine。
- [模型](https://github.com/ZhaofengWu/variable-width-transformers/blob/69dde8143d9a7912de353b57093d36d8788070d4/width_varying_model.py)：在 `WidthVaryingModel.forward` 逐层调用 `resize_if_needed`，`move_up` 从最近历史张量填回重新启用的坐标；区别于同仓的可选物理定宽残差实现。
- 原仓库的子模块路径写作 `lm-engine`，README、Python 导入和部分脚本则使用 `lm_engine`。复现前先确认路径与包导入，按 README 处理本地子模块补丁，并检查数据、保存目录及 CUDA 依赖。

## 对 wiki 的映射

- [Variable-Width Transformers 详情](../../wiki/entities/paper-variable-width-transformers.md)
