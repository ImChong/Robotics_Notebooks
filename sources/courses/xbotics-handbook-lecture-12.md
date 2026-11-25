# Xbotics 第12讲：VLA 微调实战 —— 从全量 SFT 到 LoRA 上真机 / Xbotics Lecture 12: Hands-on VLA Fine-Tuning: Full SFT, LoRA, and Robot Deployment

- **类型：** 官方教程章节 / course
- **维护者：** Xbotics 具身智能社区
- **官方章节：** https://xbotics-embodied-ai-club.github.io/Xbotics-Embodied-AI-Handbook/part3-end-to-end/12-VLA%E5%BE%AE%E8%B0%83%E5%AE%9E%E6%88%98.html
- **固定版本 Markdown：** https://github.com/Xbotics-Embodied-AI-club/Xbotics-Embodied-AI-Handbook/blob/342247fb71dd0ee15386b6eab80543c50593d3c1/docs/part3-end-to-end/12-VLA%E5%BE%AE%E8%B0%83%E5%AE%9E%E6%88%98.md
- **核对范围：** 官方章节完整正文及参考资料列表；不把课程所引第三方工作计为独立通读。
- **通读：** 已完成
- **代码入口：** https://github.com/Xbotics-Embodied-AI-club/Xbotics-Embodied-AI-Handbook/blob/342247fb71dd0ee15386b6eab80543c50593d3c1/code/vla/5_vla_finetune/5_2_smolvla_full_sft/train_sim_real_full_sft.sh
- **验证边界：** 文稿核验与源码静态走读，不代表本次重训、复现实验或真机运行。

## 编译线索

将参数更新范围与学习信号作为两个独立维度，用 SO-101 实验串起全量 SFT、LoRA SFT、多卡与仿真到真机的校验。

旧脚本未显式关闭冻结视觉塔和仅训专家的默认选项，不能凭目录名认定全模型更新；旧 ReachCube 数据与现行 PickPlaceCube40 评测也不同。新版统一任务比较展示了 loss 差距不必然形成成功率差距。

## 对 wiki 的映射

- [第 12 讲独立知识节点](../../wiki/entities/xbotics-handbook-lecture-12.md)
- [课程主节点](../../wiki/entities/xbotics-embodied-ai-handbook.md)
