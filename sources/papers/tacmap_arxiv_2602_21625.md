# Tacmap: Bridging the Tactile Sim-to-Real Gap via Geometry-Consistent Penetration Depth Map

> 来源归档（ingest）

- **标题：** Tacmap: Bridging the Tactile Sim-to-Real Gap via Geometry-Consistent Penetration Depth Map
- **简称：** Tacmap
- **类型：** paper / tactile-simulation / sim2real / dexterous-manipulation
- **arXiv abs：** <https://arxiv.org/abs/2602.21625>
- **PDF：** <https://arxiv.org/pdf/2602.21625>
- **v1 日期：** 2026-02-25（arXiv 摘要页提交历史）；v2 于 2026-05-12 修订，逐段对照 PDF 文本，v2 相对 v1 仅参考文献 [16] 的引文条目不同，正文数字一致
- **作者：** Lei Su\*、Zhijie Peng\*、Renyuan Ren、Shengping Mao、Juan Du、Kaifeng Zhang、Xuezhou Zhu（\* 共同一作；作者名单取自 arXiv 摘要页）
- **机构：** arXiv 摘要页不显示机构；论文 PDF 首页署名为 Sharpa、香港科技大学（HKUST，Zhijie Peng 双署名）、英伟达（NVIDIA，Juan Du）
- **arXiv 备注：** 8 pages；分类 cs.RO
- **项目页：** 未找到独立项目页（arXiv 摘要页、论文 PDF 均未列项目页；Sharpa 官方研究博客有一篇解读页，仅回链 arXiv，见下表）
- **代码：** 检索发现官方 GitHub 组织下的 [sharpa-robotics/sharpa-tacmap](https://github.com/sharpa-robotics/sharpa-tacmap)（README 引用本论文；Apache-2.0；初始提交 2026-04-08）。arXiv 摘要页、论文 PDF 与 Sharpa 解读页均未列此链接，是检索组织主页后得到，入库日按「已开源（仅仿真传感器子集）」处理，见下方开源核查
- **入库日期：** 2026-10-16
- **一句话说明：** 把仿真侧（物体与指尖弹性体的 3D 穿透深度）与真实侧（学习从原始触觉图像到深度图的映射）统一到同一种「deform map」几何表示，使纯仿真训练的手内旋转策略零样本迁移到 SharpaWave 真机。

## 相关资料（策展）

| 类型 | 链接 | 说明 |
|------|------|------|
| arXiv 摘要页 | <https://arxiv.org/abs/2602.21625> | 标题、作者、v1/v2 日期、摘要 |
| 官方代码（子集） | <https://github.com/sharpa-robotics/sharpa-tacmap> | Isaac Lab 触觉传感器 `SharpaTacmap` + 按压测试示例 |
| Sharpa 研究博客解读 | <https://www.sharpa.com/blogs/research/tacmap-breaking-the-sim-to-real-deadlock-in-tactile-simulation-with-a-geometric-language> | 厂商二手解读，只链 arXiv，不是项目页 |
| 同来源文章 | [机器之心 / Sharpa 微信文章归档](../blogs/wechat_sharpa_dexterous_manipulation_2026-10-16.md) | 文中仅一小段提及 Tacmap |

## 开源核查（步骤 2.5）

| 项 | 结论 |
|----|------|
| 项目页 | **无**：arXiv 摘要页与论文 PDF 均未列；Sharpa 解读页只回链 arXiv |
| 官方代码 | **部分开源**：`sharpa-robotics/sharpa-tacmap`（Apache-2.0；2026-04-08 单次初始提交）。含 `tacmap_sensor/`（`SharpaTacmap`、`SharpaTacmapCfg`，基于 Isaac Lab 的 MultiMeshRayCaster）、`env.py` / `env_cfg.py` 示例环境、`run.py` 按压测试、`assets/`（SharpaWave USD、两种压头、测试用例） |
| 未见内容 | 真实端「触觉图像 → deform map」翻译网络与自动采集台架数据、手内旋转 RL 训练代码、MuJoCo 实现；仓库内检索 resnet / ppo / checkpoint / mujoco 字样均无命中 |
| 运行验证 | 本次未运行（需 Isaac Lab 环境）；`run.py` 通过 `sys.path` 指向仓库父目录并以 `tacmap_sensor.*` 导入，据此推断仓库目录需命名为 `tacmap_sensor` |
| 综合判定 | 仿真传感器可复用，完整论文流程（真实端 + 策略训练）截至 2026-10-16 **未见公开** |

## 文章转述（微信文章，机器之心 2026-10-16）

> 以下为对文章原文的转述，标注「文章转述」，非论文原文。

- **文章转述：** Tacmap 是与英伟达合作的工作，从仿真端补齐触觉链条：把仿真与真实的触觉信号统一到同一种几何表示中，使仅在仿真中训练的手内旋转策略可以零样本迁移到真实机器人。
- **文章转述：** 文章把它放在「Sharpa 对触觉的理解经历演进」一段：早期关注多模态学习中的模态竞争（SaTA 属于这一阶段），后来认为触觉更像底层「反射弧」，应放进 System 0 做高频调整；Tacmap 补的是仿真侧。
- **文章转述：** 文章只给出论文标题与 PDF 地址，没有给出任何具体数字。

## 核心摘录（归纳，非全文）

1. **arXiv 摘要：** 触觉仿真长期面临两难——简化几何投影缺乏物理真实性，高保真 FEM 对大规模强化学习过慢。Tacmap 以体积穿透深度为锚点，仿真中把 3D 相交体积算成深度图，真实世界用自动采集台架学习从原始触觉图像到真值深度图的映射，两个域落在共享的 deform map 表示中。
2. **arXiv 摘要：** 定量评测显示仿真 deform map 与真实测量高度接近；并以手内旋转任务验证——仅在仿真中训练的策略可零样本迁移到真实机器人。
3. **文章转述：** 「把仿真与真实的触觉信号统一到同一种几何表示中，使仅在仿真中训练的手内旋转策略可以零样本迁移到真实机器人。」
4. **arXiv 摘要：** 摘要明确给出的是「定量对齐 + 零样本手内旋转」两项证据，摘要本身不含成功率数字。

## 论文正文要点（PDF v1，自行阅读；数字均为论文自报）

> 此节取自论文 PDF 正文，不属于上面三类摘录来源，单列便于核对。

- **表示：** deform map `M` 是指尖与物体之间逐像素的穿透深度；另提供净力 `F` 与接触位置 `P` 两路信号。仿真中 `F`、`P` 直接取物理引擎接触传感器；真实侧 `F` 由 ResNet 回归网络从触觉图像估计，`P` 取预测 deform map 有效接触区域的几何质心。
- **仿真几何：** 弹性体由「未形变传感面 Su」与「位于其外一个固定偏移处的虚拟感知面 Ss」界定；在 Ss 上取 H×W 网格，沿表面法向向内投射射线，`d(u,v) = max(0, zs − max(zu, zo))`，其中 `zo` 为射线与物体网格的首个交点坐标。法向局部计算使其支持曲面指尖。
- **真实侧：** 3 轴高精度运动平台驱动多种几何压头，压头 3D 位姿由编码器记录，用与仿真相同的几何投影逻辑算出真值 deform map，形成配对数据集；ResNet 编码器–解码器按逐像素 MSE 训练，从单张触觉图像预测 deform map。
- **实现：** 集成到 Isaac Lab（Raycaster API，GPU 并行）与 MuJoCo（`mj_ray`）；用 Multi-Mesh Ray Caster 预定义触觉感知点与方向，使感知分辨率与物理碰撞网格解耦。
- **硬件：** SharpaWave 灵巧手，指尖为视触觉传感器（论文称 DTC）。
- **Table I（仿真 vs 真实，各场景中位数）：** Square 压头——接触位置误差 0.66 mm、深度误差 18.53%、净力 L2 误差 0.28 N、deform IoU 88.21%；Cylinder 压头——0.96 mm、14.71%、0.61 N、85.67%。
- **效率：** 并行环境数 16 → 8192 的 GPU 显存近线性增长；开启 Tacmap 渲染后系统吞吐下降「可忽略」（文字描述，正文未给吞吐具体数值）。
- **任务：** PPO 纯仿真训练、观测含实时 Tacmap 流；零样本部署到 SharpaWave，用球形物体测试连续旋转；正文为定性描述，未给成功率。
- **局限（论文自述）：** 只建模法向穿透深度，不显式建模切向力分布（剪切应变）；物体网格越复杂，射线投射开销越大。

## 对 wiki 的映射

- [Tacmap 论文实体](../../wiki/entities/paper-tacmap.md)
- [夏帕机器人（Sharpa Robotics）](../../wiki/entities/sharpa-robotics.md)
- [Tactile Sensing（触觉感知）](../../wiki/concepts/tactile-sensing.md)
- [Sim2Real](../../wiki/concepts/sim2real.md)

## 参考来源（原始）

- arXiv：<https://arxiv.org/abs/2602.21625>
- PDF：<https://arxiv.org/pdf/2602.21625>
- 代码：<https://github.com/sharpa-robotics/sharpa-tacmap>
- 机器之心 / Sharpa 微信文章归档：[wechat_sharpa_dexterous_manipulation_2026-10-16.md](../blogs/wechat_sharpa_dexterous_manipulation_2026-10-16.md)
