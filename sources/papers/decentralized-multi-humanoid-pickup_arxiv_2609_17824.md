# 去中心化多人形搬运（arXiv:2609.17824）

> 来源归档（paper）

- **标题：** Learning Multi-Humanoid Pickup and Transport via Decentralized Object-Centric Control
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2609.17824>
- **PDF：** <https://arxiv.org/pdf/2609.17824>
- **项目页：** <https://decmht.github.io/>
- **机构：** Oregon State University（Dynamic Robotics and Artificial Intelligence Laboratory / DRAIL）
- **作者：** Helei Duan、Jeremy Dao、Jonah Siekmann、Alan Fern、Jonathan Hurst
- **入库日期：** 2026-09-20
- **核对范围：** 已提取并通读 arXiv `2609.17824` 原文全文（§I–§VII，2,749 行提取文本），并打开项目页 `https://decmht.github.io/` 核对链接。
- **通读：** 已完成
- **一句话说明：** 每机基于物体附着点参考的同一局部观测 PPO 策略（零机间通信），从单机拾取零样本泛化至十机共搬、搬运交接及两台实体 Digit V3 协同搬箱。

## 开源状态

- **待发布**（步骤 2.5 复核 2026-11-29：项目页 `decmht.github.io` 返回 HTTP 200，仅列 arXiv/PDF/HTML 与演示视频，未附 GitHub 代码或模型权重链接）

## 核心摘录

1. **策展来源：** [senlanke 周更](../../sources/blogs/wechat_senlanke_weekly_humanoid_quadruped_2026-09-14_18.md)
2. **Object-Centric Attachment Framing：** 通过相对箱体几何定义的局部附着点（local attachment frame）与相对偏移特征将任意团队规模、不同物体外形及异构负载分配转化为单机一致的局部观测空间，训练与部署阶段均无需机间显式通信。
3. **团队规模与异构力分配泛化：** 训练仅覆盖至最多 4 机场景，测试零样本泛化到 1–10 台 Digit 人形机器人在平地、斜坡（4°–8°）、楼梯、2–6° 无规划急转、异质载荷分担及顺序递送交接（relay handoff），并消融验证了局部载荷感知对抵抗侧倾力矩的作用。
4. **Sim2Real（两台实体 Digit V3）：** 在不进行真实多机联合微调的前提下迁移到两台 Agility Digit V3 实体机器人上协同拾起、抬行与放下不同质量/几何物体。

**对 wiki 的映射**

- [paper-decentralized-multi-humanoid-pickup](../../wiki/entities/paper-decentralized-multi-humanoid-pickup.md)
