# Domain Randomization (Tobin et al. 2017) 通读

> 来源归档（原文通读 · 支撑 `wiki/concepts/sim2real.md`）

- **标题：** Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World
- **作者：** Josh Tobin, Rachel Fong, Alex Ray, Jonas Schneider, Wojciech Zaremba, Pieter Abbeel（OpenAI / UC Berkeley）
- **类型：** paper / domain-randomization / sim2real
- **原文：** <https://arxiv.org/abs/1703.06907>（v1，2026-11 通过 arXiv PDF 获取全文）
- **一句话说明：** 域随机化奠基工作：仅用低保真、非真实纹理的仿真渲染图训练物体定位网络，零真机数据迁移，真机精度约 1.5 cm，并用于杂乱场景抓取。
- **核对范围：** 通读摘要、§I 引言、§II 相关工作、§III 方法、§IV 实验（定位精度、消融、机器人抓取）、§V 结论；参考文献仅浏览。
- **通读：** 已完成

## 核心摘录（均出自原文）

- **假设（§I）：** 若仿真中的变化足够大，真实世界对模型而言只是又一种变化，则无需额外真机训练即可泛化。作者列举现实差距来源：系统辨识耗时易错、未建模效应（非刚性、齿轮间隙、磨损、流体）、低保真传感器渲染。
- **随机化内容（§III-A）：** 干扰物数量与形状（0–10 个）、物体 / 桌面 / 地面 / 天空盒 / 机器人纹理、相机位置（初始点周围 10×5×10 cm）、朝向（±0.1 rad）与视场（±5%）、光源数量与位置与镜面特性、图像噪声；纹理为随机 RGB、渐变或棋盘格，用 MuJoCo 内置渲染器，无需照片级真实。
- **模型与训练（§III-B）：** 改造的 VGG-16，L2 损失、Adam，学习率约 1e-4。ImageNet 预训练并非必需：数据足够时随机初始化几乎同样好。
- **结果（§IV-B）：** 8 种几何体、480 张真机图像，平均定位误差在 1.5 cm 内，含干扰物与部分遮挡；仍对仿真数据过拟合（仿真误差 0.3–0.5 cm）。
- **消融（§IV-C）：** 唯一纹理数少于约 1000 时性能显著下降；训练中不放干扰物，真机有干扰物时误差从约 1.8 升至 7.2 cm；相机位置随机化带来小幅提升；加噪声影响可忽略；训练样本约 5000 起可用、约 50000 以内持续提升。
- **机器人实验（§IV-D）：** Fetch 机器人配合现成运动规划器，在杂乱场景 40 次中成功抓取 38 次；YCB 的 Spam 罐 10 次中成功 9 次。
- **局限与展望（§V）：** 精度仍不足以做接触丰富或高精度操作；文中提及无法在真机上做验证（训练时假设无真实数据）；方向包括更高分辨率、立体 / 深度、与域适应结合。

## 对 wiki 的映射

- [`wiki/concepts/sim2real.md`](../../wiki/concepts/sim2real.md)

## 参考来源

- 论文：<https://arxiv.org/abs/1703.06907>
