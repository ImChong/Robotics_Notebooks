# 「具身智能扎堆流匹配」：Flow‑Matching全面取代Diffusion了？

> 来源归档（blog / 微信公众号）

- **标题：** 「具身智能扎堆流匹配」：Flow‑Matching全面取代Diffusion了？
- **类型：** blog
- **作者：** 深蓝具身智能（编辑｜咖啡鱼；审编｜具身君；《具身智能基础》专栏第 14 篇）
- **原始链接：** https://mp.weixin.qq.com/s/Reqadr6Jpp6a1CP9TlCTUw
- **发表日期：** 2026-09-26（入库日）
- **入库日期：** 2026-09-26
- **抓取方式：** WebFetch（Camoufox 工具链未预装于本环境）
- **原始抓取落盘：** [`sources/raw/wechat_shenlan_flow_matching_embodied_column14_2026-09-26.md`](../raw/wechat_shenlan_flow_matching_embodied_column14_2026-09-26.md)
- **专栏专辑：** [《具身智能基础》](https://mp.weixin.qq.com/mp/appmsgalbum?__biz=MzkwMDcyNDUzMQ==&action=getalbum&album_id=4525948187102363653)
- **一句话说明：** 科普流匹配：高维「数据点」在归一化时间向量场中 ODE 积分生成轨迹/图像；对比 Diffusion 的随机往复；具身侧 π₀/π₀.5、GR00T N1、SmolVLA 等动作用 FM 专家；训练四环节与 CPU/GPU 算力分工。

## 文中系统 / 论文 → 本库已有详情节点

| 引用 | 说明 | wiki（不新建重复 arXiv 节点） |
|------|------|--------------------------------|
| π₀ / π₀.5 | Physical Intelligence 流匹配动作专家 | [π₀ 方法页](../../wiki/methods/π0-policy.md)、[π₀.7 方法页](../../wiki/methods/pi07-policy.md)、[paper-pi0](../../wiki/entities/paper-pi0.md) |
| GR00T N1 | NVIDIA 论文写明 flow matching 动作生成 | [paper-hrl-stack-34-gr00t_n1](../../wiki/entities/paper-hrl-stack-34-gr00t_n1.md) |
| SmolVLA | HF 0.45B，动作专家用流匹配 | 底座 [arXiv:2506.01844](https://arxiv.org/abs/2506.01844)；部署实例 [paper-ros2smolvla](../../wiki/entities/paper-ros2smolvla.md) |

## 对 wiki 的映射

- **概念编译页（新建）：** [flow-matching-embodied-policy](../../wiki/concepts/flow-matching-embodied-policy.md)
- **形式化交叉：** [probability-flow](../../wiki/formalizations/probability-flow.md)
- **系统课对照：** [MIT 6.S184 Flow Matching & Diffusion](../../wiki/overview/mit-flow-matching-diffusion-2026.md)
