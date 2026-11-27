# Tencent/WeVisOne-VL

- **类型：** 官方仓库
- **机构：** 腾讯微信视觉团队
- **代码：** <https://github.com/Tencent/WeVisOne-VL>
- **版本：** [80ecbdd](https://github.com/Tencent/WeVisOne-VL/tree/80ecbdddb9526727c312f2c837d97e0ddf04e01f)，2026-11-24 北京时间；2026-11-26 开放范围核对使用该版本
- **权重：** [2B](https://huggingface.co/tencent/WeVisOne-VL-2B) · [4B](https://huggingface.co/tencent/WeVisOne-VL-4B)
- **许可证：** 仓库 Apache-2.0；第三方项目、组件和数据依各自条款
- **通读：** 已完成
- **核对范围：** 固定提交 README、推理与像素解码主路径、评测入口、训练 README、SFT/GRPO 启动和奖励源码；未训练、未跑权重或基准。

## 开放时间线与范围

- 10 月公开推理包、Cookbooks、评测流水线、2B/4B 权重及 Gradio 演示；[Mask2Alpha](https://github.com/Tencent/WeVisOne-VL/tree/mask2alpha) 是独立的抠图标注工具分支，不是完整 WDE。
- 11 月 23 日[公开 SFT 和 GRPO 微调代码](https://github.com/Tencent/WeVisOne-VL/commit/8fe7e9dc538937d813a89693d1fb6055ae8ed877)；11 月 24 日[修订框匹配奖励按 IoU 三次方计分](https://github.com/Tencent/WeVisOne-VL/commit/f700ff8760bf6fba529bdcaf1b246e8b9066eef0)。旧记录「微调待办」已不适用。核查版本有格式/配置示例，不含论文完整 WDE 语料及全量数据流水线。
- [training/README.md](https://github.com/Tencent/WeVisOne-VL/blob/80ecbdddb9526727c312f2c837d97e0ddf04e01f/training/README.md)说明从仓库根目录用自备多模态 JSONL 做 SFT；可选训练像素解码器，默认关闭。GRPO 将同格式 JSONL 转为 verl parquet，微调策略但不训练像素解码器，脚本把起点权重的 `mask_decoder.pt` 复制进导出的 HF 目录。
- [engine.py](https://github.com/Tencent/WeVisOne-VL/blob/80ecbdddb9526727c312f2c837d97e0ddf04e01f/wevisone_vl/engine.py) 的 `predict` 先生成文本，再在要求解码且出现 `<OBJ>` 时按框解析进入 `decode_masks`。后者把预测答案重新随提示前向，取 `<OBJ>` 对应隐藏状态和多层视觉特征，调用像素解码器。
- [评测目录](https://github.com/Tencent/WeVisOne-VL/tree/80ecbdddb9526727c312f2c837d97e0ddf04e01f/evaluation)将样本准备、推理和各任务指标区分，代码存在不等于所有论文内部基准和样本已开放。

## 最小运行检查

CPU 直接调用该提交的 `parse_seg_boxes` 和 `training.rl.reward.compute_score`，8 项合成输入断言通过。方框仅在像素任务要求的合法 `<OBJ>` 配对时进入解码；单框 IoU=0.5 的奖励为 0.125。在负样本上，合法 `none` 与没有可解析对象的畸形文本都可能取得 1 分，这是该奖励实现检查空集合而非严格语法的结果。不等于验证模型实际拒答率。

- [知识页](../../wiki/entities/paper-wevisone-vl.md)
- [论文归档](../papers/wevisone_vl_arxiv_2610_31329.md)
