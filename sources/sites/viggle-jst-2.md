# Viggle JST-2（官方项目资料归档）

> 来源归档（site / project-page）

- **标题：** Introducing JST-2（Joint Space-and-Time）
- **类型：** 官方项目发布页 / research pages / 产品与 API
- **机构：** Viggle
- **官方入口：** <https://viggle.ai/>
- **场景 / 世界模型：** <https://viggle.ai/research/jst2-scenes>
- **角色基准：** <https://viggle.ai/research/jst2-character>
- **动作基准：** <https://viggle.ai/research/jst2-motion>
- **API：** <https://viggle.ai/developers>
- **入库日期：** 2026-10-21
- **一句话说明：** Viggle 将 JST-2 描述为 Joint Space-and-Time 世界模型，在共享三维表示中生成、重建并扩展场景，也生成可动角色与全身动作。

## 官方资料要点

- 官方将 JST-2 定位为物理原生世界基础模型。其发布介绍把场景、角色、动作置于共享 3D 世界中，并称新场景可在数秒内生成和扩展。
- 官方把世界表示为具有持续身份的 3D atoms；Physics-Native Tokens 包含显式 xyz 位置以及学习到的属性和动力学向量。编码器可由图像推断这些 token，生成器也可从多模态提示产生 token，解码器据视角、运动或力输入生成图像或 3D Gaussian splats。
- 官方场景页称模型支持最多 64 个视角作为空间上下文；官方发布介绍称在 NVIDIA B200 上生成场景约 1 秒、推理成本低于 0.01 美元，计时不含网络延迟。这些是厂商报告值，不是独立复现结果。
- 角色页说明可由文本或图像生成带动作动态的角色；动作页说明可由文本或视频生成全身 3D 动作，并应用到 JST-2 角色或兼容标准角色骨架的资产上。
- 官方提出用稀疏真实观测重建并扩展环境，可用于机器人仿真环境构建等方向；这描述的是潜在用途，不等于已验证的可训练物理仿真。

## 开源核查（步骤 2.5）

- **截至 2026-10-21，项目发布页、研究页和开发者页没有列出 JST-2 的公开训练 / 推理源码仓库、权重或数据集链接。**
- **模型状态：** 通过 Viggle 的 PINOC 产品和 API 提供；不能据公开资料判断为可本地部署或开源模型。
- **可访问入口：** [PINOC](https://viggle.ai/) 产品体验与 [Viggle API](https://viggle.ai/developers)。API 是托管服务，不代表模型源码开放。

## 对 wiki 的映射

- [Viggle JST-2 实体页](../../wiki/entities/viggle-jst-2.md)
- [Viggle 官方项目页 / 场景说明](https://viggle.ai/)
- [角色基准](https://viggle.ai/research/jst2-character)
- [动作基准](https://viggle.ai/research/jst2-motion)
- [API 文档入口](https://viggle.ai/developers)
