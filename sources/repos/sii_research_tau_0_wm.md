# sii-research/tau-0-wm

> 来源归档

- **标题：** τ₀-World Model（官方实现）
- **类型：** repo
- **组织：** sii-research（Shanghai AI Innovation Institute 等，以 upstream 为准）
- **代码：** <https://github.com/sii-research/tau-0-wm>
- **项目页：** <https://finch.agibot.com/research/tau0-wm>
- **技术报告：** <https://finch-static.agibot.com/VAM/blog/tau_0_wm.pdf>
- **权重：** <https://huggingface.co/sii-research/tau-0-wm>（Apache-2.0；VAM 扩散权重已发布）
- **入库日期：** 2026-05-31
- **一句话说明：** τ₀-WM 官方仓库：基于 **Wan-2.2 TI2V-5B** 的 VAM 部署服务（openpi 风格 WebSocket policy server）、双臂 EEF 绝对位姿动作空间说明；**Simulator 权重** 与 **测试时计算** 代码待后续发布。

## 与本仓库知识的关系

| 主题 | 关系 |
|------|------|
| [τ₀-World Model](../../wiki/entities/tau0-world-model.md) | 实体归纳页：异构监督、双接口、测试时闭环 |
| [mimic-video（VAM）](../../wiki/methods/mimic-video.md) | 同属 Video-Action Model 命名；mimic-video 偏互联网视频潜计划 + 流匹配动作头，τ₀-WM 强调 **联合骨干 + 动作条件仿真 + 测试时搜索** |
| [GE-Sim 2.0](../../wiki/entities/ge-sim-2.md) | 同 Agibot 系视频世界模型栈；GE-Sim 2.0 侧重 **闭环模拟器 + World Judge**，τ₀-WM 把 **策略与后果评估** 收进同一 5B 表征 |
| [World Action Models](../../wiki/concepts/world-action-models.md) | Joint WAM 族实例：共享预测表征同时服务控制与想象 |

## 部署要点（README 摘要，以克隆时 upstream 为准）

- **依赖：** `pip install -r requirements.txt`；另需本地 **Wan2.2-TI2V-5B** 与 τ₀-WM 权重，在 `configs/deployment/wan_pretrain_rela_eef6d.yaml` 配置路径。
- **状态输入：** 双臂 EEF **绝对位姿** 14 维（xyz + 四元数 xyzw，原点为各臂 base link）；夹爪 2 维（0–120）。
- **动作输出：** 形状 `{T, 16}` — 左右 EEF 绝对位姿 + 归一化夹爪开合（0–1）；预训练内部为 **相对位姿 + 6D 旋转**（20 维），四元数↔6D 自动转换。
- **服务：** `bash run_infer_server.sh $HOST $PORT`；客户端示例 `web_infer_utils/simple_client.py`。
- **致谢栈：** [Wan-2.2](https://github.com/Wan-Video/Wan2.2)、[GE-Act / Genie-Envisioner](https://github.com/AgibotTech/Genie-Envisioner)、[openpi](https://github.com/Physical-Intelligence/openpi) WebSocket 服务模式。

## 对 wiki 的映射

- 沉淀 **[`wiki/entities/tau0-world-model.md`](../../wiki/entities/tau0-world-model.md)**；技术报告摘录见 [`sources/papers/tau0_wm_tech_report.md`](../papers/tau0_wm_tech_report.md)。
