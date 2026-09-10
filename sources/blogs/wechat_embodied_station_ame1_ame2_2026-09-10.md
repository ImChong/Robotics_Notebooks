# 从「下一脚看哪里」到「怎样穿过整片地形」：AME-1 到 AME-2 改变了什么？

> 来源归档（blog / 微信公众号）

- **标题：** 从「下一脚看哪里」到「怎样穿过整片地形」：AME-1 到 AME-2 改变了什么？
- **类型：** blog
- **作者：** 具身智能之心（微信公众号）
- **原始链接：** https://mp.weixin.qq.com/s/VU_JcNP2FZrITTUoA8IDuA
- **发表日期：** 2026-09-10（入库日）
- **入库日期：** 2026-09-10
- **抓取方式：** WebFetch（桌面 UA 返回微信验证页；正文由 WebFetch 可读通道获取）
- **原始抓取落盘：** [`sources/raw/wechat_embodied_station_ame1_ame2_2026-09-10.md`](../raw/wechat_embodied_station_ame1_ame2_2026-09-10.md)
- **一句话说明：** 对照 AME-1（Science Robotics 2025）与 AME-2（arXiv:2601.08485）在 query 条件化、神经映射、任务接口与 Teacher–Student 上的系统级演进；**2/2 独立详情节点，0 重复 arXiv，复用既有实体**。

## 文内点名 → 本库节点

| # | 资料 | 身份 | 开源结论（入库日） | wiki |
|---|------|------|-------------------|------|
| 01 | Attention-Based Map Encoding（AME-1） | 论文 arXiv:2506.09588 / *Science Robotics* | **官方训练代码未发布**；Zenodo 数据；社区 [SII-FUSC/AME_Locomotion](https://github.com/SII-FUSC/AME_Locomotion) **非官方** G1 复现 | [paper-ame-attention-based-map-encoding](../../wiki/entities/paper-ame-attention-based-map-encoding.md) **复用** |
| 02 | AME-2: Agile and Generalized Legged Locomotion | 论文 arXiv:2601.08485 | **官方训练代码未发布**；项目页无 GitHub；社区 [Kitjesen/ame2](https://github.com/Kitjesen/ame2) **非官方** ANYmal-D PyTorch 复现 | [paper-notebook-ame-2-agile-and-generalized-legged-locomotion-vi](../../wiki/entities/paper-notebook-ame-2-agile-and-generalized-legged-locomotion-vi.md) **复用** |

## 文内要点速记

1. **问题升级：** AME-1 答「下一脚看哪里」；AME-2 答「怎样穿过整片混合地形」——需全局语境 + 长期地图记忆 + 不确定性。
2. **Query 条件化：** proprio-only → **global ∥ proprio** 联合 query local map tokens（消融支撑）。
3. **感知—行动闭环：** 给定 elevation map → **在线神经映射**（高度+方差、Probabilistic WTA 融合）；Student 训练栈 = 部署栈。
4. **任务接口：** **速度跟踪** → **目标到达**；允许减速、侧身、重试等中间自由度（跑酷/全身接触涌现前提）。
5. **训练范式：** 两阶段 PPO（理想→噪声）→ **Teacher（GT 地图）→ Student（在线映射）** 动作+表征蒸馏 + PPO。
6. **量化差距（文内表）：** 训练地形均 >90%；稀疏 Test1 AME-1 **99.2%** vs AME-2 teacher **96.8%**；**四组未见混合地形均值** AME-1 **51.2%** vs teacher **95.2%**、student **82.4%**。
7. **边界：** 2.5D 高程、里程计/静态假设、相机关闭时主动感知退化；缺大规模重复试验统计。

## 对 wiki 的映射

- **2/2 独立详情节点**；**0 重复 arXiv 节点**。
- 阅读坐标：[AME-1→AME-2 技术地图](../../wiki/overview/ame-1-to-ame-2-technology-map.md)
- 交叉：[Terrain Adaptation](../../wiki/concepts/terrain-adaptation.md)、[Privileged Training](../../wiki/concepts/privileged-training.md)、[楼梯与障碍 Locomotion](../../wiki/tasks/stair-obstacle-perceptive-locomotion.md)

## 当前提炼状态

- [x] 公众号正文抓取与 raw 归档
- [x] 2 篇独立节点核查（**2 复用 / 0 新建 / 0 重复 arXiv**）
- [x] 项目页与社区仓开源状态核查（步骤 2.5）
