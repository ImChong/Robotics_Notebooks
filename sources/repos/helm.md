# HeLM 官方代码

> 来源归档（repo）

- **标题：** HeLM
- **类型：** repo
- **链接：** https://github.com/kimspacehuni/HeLM
- **论文：** https://arxiv.org/abs/2611.02660
- **项目页：** https://kimspacehuni.github.io/HeLM/
- **许可证：** Apache-2.0（官方 README）
- **入库日期：** 2026-11-24
- **核对范围：** README、eval_real_time_main.py 及 eval_real_time_qwen.py、eval_real_time_pi0.py 的关键接口；未运行真机。
- **开放状态：** 数据处理、训练、推理脚本及 51 个任务配置公开；完整评测轨迹未发布于该仓库。

## 复现入口

- 数据：`helm_datasets.extract_frames` → `annotate_app.py` → `helm_datasets.build_helm` → `helm_datasets.merge_helm_data`。
- 高层训练：`train/train_helm/train_helm.py`，低层：`train/train_llp/train_pi0.py`。
- 高层验证：`evaluate/eval_helm/eval_helm_hlp.py`。
- 真机入口：[eval_real_time_main.py](https://github.com/kimspacehuni/HeLM/blob/main/evaluate/eval_HLP_LLP/eval_real_time_main.py)，需指定 TaskSpec、HLP 基座与 adapter、LLP checkpoint、数据路径和设备配置。
- [Qwen 高层接口](https://github.com/kimspacehuni/HeLM/blob/main/evaluate/eval_HLP_LLP/eval_real_time_qwen.py) 与 [π0 低层接口](https://github.com/kimspacehuni/HeLM/blob/main/evaluate/eval_HLP_LLP/eval_real_time_pi0.py)。

## 相关归档

- [论文](../papers/helm-arxiv-2611-02660.md)
- [项目页](../sites/helm.md)
- [知识实体](../../wiki/entities/paper-helm-2611-02660.md)
