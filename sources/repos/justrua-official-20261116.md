# JustRUA: A Controlled Study of GPT-6 Robot-Use Agents — 公开仓库运行入口核对

- **仓库：** https://github.com/ZefanW/JustRUA
- **日期：** 2026-11-16
- **开放状态：** 公开了完整的 XPolicyLab policy adapter；依赖 RoboDojo 仿真/资产、Depth Pro checkpoint 与 GPT-6 Astra 服务账户。
- **对应论文：** [arXiv:2611.16009](https://arxiv.org/abs/2611.16009)
- **项目页：** https://zefanw.github.io/JustRUA/
- **原文归档：** [论文](../papers/justrua-arxiv-2611-16009.md)

## 已核对的实际运行路径

- `policy/JustRUA/deploy.py` 将 `Observer`、`CodingClient`、`JustRUAPolicy` 接入 `XPolicyLab` 的 Direct EEF runner。
- `policy/JustRUA/policy.py` 中 `CONTROL_CAP=20`，截断控制序列并用 `confirm_executed` 恢复实际夹手目标与执行反馈。
- `policy/JustRUA/README.md` 中有安装、`eval.sh` 和离线 debug transport 测试命令；需要外部 RoboDojo、XPolicyLab、Depth Pro 权重与模型服务。
