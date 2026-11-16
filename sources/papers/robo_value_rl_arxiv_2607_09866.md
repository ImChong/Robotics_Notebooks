# Robo-ValueRL：可靠价值估计与离线到在线强化学习

- **标题：** Robo-ValueRL: Reliable Value Estimation for Offline-to-Online Reinforcement Learning
- **论文：** https://arxiv.org/abs/2607.09866
- **原文 HTML：** https://arxiv.org/html/2607.09866
- **版本：** arXiv v1（2026-07-10）
- **项目页：** https://gewu-lab.github.io/Robo-ValueRL/
- **代码：** https://github.com/Open-X-Humanoid/Robo-ValueRL
- **模型与数据：** https://huggingface.co/collections/X-Humanoid/robo-valuerl
- **机构：** 中国人民大学高瓴人工智能学院、北京人形机器人创新中心、北京林业大学、北京大学、微软亚洲研究院
- **核对范围：** 通读 arXiv v1 HTML：摘要、引言、相关工作、§3.1–3.4 方法、§4.1–4.4 结果、结论及实现附录 §7–11；对照官方 README 与训练、RTC 客户端源码核查复现入口。未实际训练或部署。
- **通读：** 已完成

## 机制与实验依据

1. 历史条件价值估计器预测剩余任务时间，减轻部分可观测及遮挡导致的单帧进度歧义；价值可靠性同时采用全局进度与局部动作偏好指标核验。
2. 在离线示教上以预测的进度变化对 action chunk 排序，再转换为 GOOD/MEDIUM/BAD 质量标签；训练质量条件的 consistency VLA 策略，避免把全部次优片段同等对待。
3. 真机阶段通过 socket 将机器人观测传给 GPU 策略服务器、RTC 执行动作块；采集带人工干预的 rollout，用在线残差适配器纠正易出错步骤，保持离线策略先验。
4. 芯片插入以严格质量筛选最佳，3 次在线迭代、每次 500 条轨迹，从 **46%** 上升至 **86%** 成功率；积木拆解最后报告 **84%**，其中较宽松筛选更合适（§4.2–4.3）。
5. 局限：价值误估会错误删掉有效抓取或粗定位样本；任务依赖严格/宽松阈值；训练主要依赖同一真机及人工干预，跨本体泛化不应直接假设。

## 复现入口（官方仓库）

- `robo_valuerl/scripts/train_robo_valuerl_value_estimator.sh` → `train_robo_valuerl_value_estimator.py`
- `robo_valuerl/annotate_value_function_sparse_part.py` → `generate_action_quality/generate_quality.py`
- `robo_valuerl/scripts/train_robo_valuerl_offline.sh` → `train_robo_value_rl_offline_rl.py`
- `robo_valuerl/deploy_robo_valuerl.py`（GPU）↔ `rtc_env_client_inference_with_delta.py`（机器人）
- `robo_valuerl/scripts/train_robo_valuerl_online.sh` → `train_robo_value_rl_online_rl.py`

## 对 wiki 的映射

- [Robo-ValueRL](../../wiki/entities/paper-robo-valuerl.md)
- [SafeDojo：训练期约束安全](../../wiki/entities/paper-safedojo.md)
