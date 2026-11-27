# iMFPO 官方实现

- **类型：** 源码
- **链接：** https://github.com/AIGeeksGroup/iMFPO
- **核对提交：** [cad0d8c](https://github.com/AIGeeksGroup/iMFPO/tree/cad0d8cc80fd247c6c42e1f59c0fbd3a1dedf7c4)
- **许可证：** Apache-2.0（仓库 LICENCE）
- **论文：** [arXiv:2611.33922v1](../papers/imfpo_arxiv_2611_33922.md)
- **项目页：** [官方演示](../sites/imfpo.md)
- **模型：** https://huggingface.co/AIGeeksGroup/iMFPO

训练入口 `scripts/run_ppo_teacher.sh` 与 `scripts/run_imfpo.sh`，策略实现与 runner 在 `source/isaaclab_fpo/`，机器人环境在 `source/unitree_rl_lab/`。核对 `scripts/fpo/eval_imfpo.py`、`scripts/fpo/export_onnx.py` 以及 `ActorCritic.act_inference` / `meanflow_one_step`。环境、部分机器人资产和真机部署栈需另行准备。代码核查为静态阅读，未运行训练或真机。

官方权重快照 [e28f474b](https://huggingface.co/AIGeeksGroup/iMFPO/tree/e28f474b100927d17af8b9409bce42c61fc14cd9) 的 [部署配置](https://huggingface.co/AIGeeksGroup/iMFPO/blob/e28f474b100927d17af8b9409bce42c61fc14cd9/deploy.yaml) 声明 45 维观测、12 关节动作、20 ms 周期。对同快照的 ONNX 使用 ONNX Runtime CPU 跑零值与常量 0.1 的合成观测，得到有限的 float32 [1, 12] 输出；仅验证文件与 I/O，未运行仿真或真机。
