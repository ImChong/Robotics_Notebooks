# 具身视觉语言模型驱动的崎岖地形交互导航（2611.11665）

> 论文一手原文归档

- **英文题目：** HINT-VLM: Humanoid Interactive Navigation over Rough Terrain with Embodiment-Grounded Vision-Language Model
- **arXiv：** https://arxiv.org/abs/2611.11665
- **HTML 全文：** https://arxiv.org/html/2611.11665v1
- **PDF：** https://arxiv.org/pdf/2611.11665v1
- **机构：** 佐治亚理工学院、VinMotion、南加州大学
- **官方项目：** https://yzwfromk.github.io/HINT_VLM/
- **入库日期：** 2026-11-16
- **核对范围：** 通读并核对 arXiv v1 III VLM/GP 校准、IV 目标掩码/地形补全、V 规划和状态机、VI 表 II 场景与消融、VII 结论、项目页 Code 状态；与用户提供的 29 篇 PDF 周报核对研究问题、实验口径与局限。另查官方项目页的开放状态。
- **通读：** 已完成
- **一句话说明：** VLM 提供物体可移动性先验，交互数据训练 Gaussian Process 残差校正该机体的推物成功率；Grounding DINO/SAM2 掩码和深度投影至 BEV，LaMa 补全被挡地面；结合 STATE-NAV 通行估计与交互成功/失败代价，选择推、绕行或放弃。

## 原文摘录（提炼）
1. **问题与动机：** 物体可以移动，不代表机器人一定能推动，更不代表物体下面被遮挡的地面可安全通过。
2. **方法：** VLM 提供物体可移动性先验，交互数据训练 Gaussian Process 残差校正该机体的推物成功率；Grounding DINO/SAM2 掩码和深度投影至 BEV，LaMa 补全被挡地面；结合 STATE-NAV 通行估计与交互成功/失败代价，选择推、绕行或放弃。
3. **实验：** 三场景 Garden、Construction、Park 的仿真导航成功率 95.5%、97.7%、100%；Park 移除遮挡地形重建后为 86.4%。官方项目页展示室内推车和户外花园定性真机视频，但 Code 标为 Coming Soon。
4. **边界与局限：** 隐藏地形来自补全估计而非真实观测；真机视频不能替代三场景仿真统计。项目页截至 2026-11-16 暂无可运行代码。

## 开源状态
官方项目页明确 Code Coming Soon，缺少可运行入口。

## 对 wiki 的映射
- [HINT-VLM](../../wiki/entities/paper-hint-vlm-rough-terrain.md)
