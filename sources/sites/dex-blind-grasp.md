# Blind Dexterous Grasping — 官方项目页

> 来源归档（官方项目网站）

- **标题：** Blind Dexterous Grasping via Real2Sim2Real Tactile Policy Learning
- **项目：** <https://dex-blind-grasp.github.io/>
- **论文：** <https://arxiv.org/html/2606.11767>（v2，2026-06-11）
- **机构：** 上海科技大学（ShanghaiTech University）；北京通用人工智能研究院（BIGAI）
- **平台：** xArm6（6 DoF）+ LEAP Hand（16 DoF）+ 44 路二值触觉（4×8 TwinTac + 12 FSR）
- **页面核验：** 2026-11-23
- **核对范围：** 项目页摘要、方法流程、硬件说明、实验结果、局限及代码/硬件入口。
- **通读：** 已完成

## 开源核查

- **代码：** 项目页的 Code 链接当前是站内占位入口，未指向可检查的官方实现仓库；不能声称训练/推理代码已发布。
- **硬件：** 页内称将提供传感器 PCB、打印件和 ROS2 驱动，但「Open-source hardware repository (coming soon)」同样没有公开仓库链接。
- **复现状态：** **待发布**；原文 Appendix E–G 给出算法和参数描述，但不等同可运行源码。
- **核验时间：** 2026-11-23。

## 一手内容要点

1. 真机采集 tapping/sliding 标定动作，回放到数字孪生仿真，拟合二值触觉事件的接触阈值与指尖串扰模型。
2. 成功抓取示范来自仿真中针对每种对象训练的 PPO 专家；特权信息只在训练和触觉编码器预训练阶段使用。
3. 触觉位置（FK 坐标）与二值激活进入时序 Transformer，输出特征供 Diffusion Policy 生成动作块。
4. 真机 20 个对象、每个 5 次试验，总成功率 27/100；**并非**仅凭关节本体感知，部署仍使用稀疏二值触觉。

## 对 wiki 的映射

- [Blind Dexterous Grasping 实体页](../../wiki/entities/paper-sa-2606-11767-blind-dexterous-grasping-via-real2sim2real-tacti.md)
- [论文一手归档](../papers/sun_awesome_r2s2r_2606_11767_blind-dexterous-grasping-via-real2sim2re.md)
