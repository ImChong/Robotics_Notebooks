# 曾让大模型后训练变简单、变便宜的 pi 联创，如今觉得：快轮到机器人了。

> 来源归档（blog / 微信公众号）

- **标题：** 曾让大模型后训练变简单、变便宜的 pi 联创，如今觉得：快轮到机器人了。
- **类型：** blog
- **作者：** 具身智能之心（微信公众号）
- **原始链接：** <https://mp.weixin.qq.com/s/tfoY3_yF9Ia5qJFYCao6cw>
- **发表日期：** 2026-09-29（入库日推断）
- **入库日期：** 2026-09-29
- **抓取方式：** WebFetch（正文完整）
- **一句话说明：** 以 Chelsea Finn（PI 联创、DPO 共同作者）与 Perry Dong 博客为轴，串联 **DPO→LLM 后训练配方→机器人 VLA 可靠性缺口→EXPO/EXPO-FT 真机 30/30 与 19 min 在线数据**，并强调 reward/reset/HIL 协议尚未标准化。

## 核心摘录（归纳，非全文）

### 1) 叙事主线

- 机器人「会做」≠「敢走开」；问题在 **post-training** 如何把已有能力推到部署级 nines。
- Finn 从 **DPO**（简化 RLHF、Llama 3 采用）类比到机器人：**强预训练 VLA + 缺通用后训练配方**。
- 三连读：[post-training 博客](https://pd-perry.github.io/posts/post-training.html)、[EXPO arXiv:2507.07986](https://arxiv.org/abs/2507.07986)、[EXPO-FT arXiv:2605.25477](https://arxiv.org/abs/2605.25477)。

### 2) EXPO-FT 数字（公众号转述论文）

- 插花：SFT 基线 **14/30** → EXPO-FT **30/30**（在线交互 **14 min**）。
- 八项任务平均 **19.1 min** 在线交互、每项 **30/30**（台球、插花、灯串等）。
- 机制转述：**大 VLA 提案 + 小编辑策略 + Q 比较 + 成功轨迹回灌**（源自 EXPO 系）。

### 3) 开放问题（转述博客）

- **Reward / Reset / HIL** 无 LLM RLVR 级默认；超参、初始化仍 craft。

## 对 wiki 的映射

- [universal-post-training-robotics](../../wiki/concepts/universal-post-training-robotics.md)
- [paper-expo](../../wiki/entities/paper-expo.md)
- [paper-expo-ft](../../wiki/entities/paper-expo-ft.md)
- [paper-real-time-expo-ft](../../wiki/entities/paper-real-time-expo-ft.md)

## 参考来源（原始）

- 微信公众号：<https://mp.weixin.qq.com/s/tfoY3_yF9Ia5qJFYCao6cw>
