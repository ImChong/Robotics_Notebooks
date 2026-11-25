# Berkeley Humanoid Lite: An Open-source, Accessible, and Customizable 3D-printed Humanoid Robot

> 来源归档（ingest · Robot Learning Paper Notebooks progress 待深读）

- **标题：** Berkeley Humanoid Lite: An Open-source, Accessible, and Customizable 3D-printed Humanoid Robot
- **类型：** paper
- **深读状态：** 待撰写（见 [papers/PROGRESS.md](https://github.com/ImChong/Robot_Learning_Paper_Notebooks/blob/main/papers/PROGRESS.md)）
- **计划笔记路径：** `papers/12_Hardware_Design/berkeley-humanoid-lite-an-open-source-accessible/berkeley-humanoid-lite-an-open-source-accessible.md`
- **分类：** 12_Hardware_Design
- **arXiv：** <https://arxiv.org/abs/2504.17249>
- **项目页：** <https://lite.berkeley-humanoid.org/>
- **代码：** <https://github.com/HybridRobotics/Berkeley-Humanoid-Lite>
- **入库日期：** 2026-06-11（2026-11-21 完成 `arXiv:2504.17249v1` 全文通读与 BOM/实验表核对）
- **通读：** 已完成
- **一句话说明：** 已完成 `arXiv:2504.17249v1`（UC Berkeley Hybrid Robotics）全文通读与实体页面归纳同步。

## 核心摘录（原文 PDF 精读实证）

- **整机架构与 BOM（§III, Table I–III）**：高 **0.8 m**、重 **16 kg**，共 22 个模块化执行器（**10× 6512** + **12× 5010**）与桌面 FDM-PLA 3D 打印结构（全件适应 200×200×200 mm 打印体积），6512 含 MAD M6C12 150KV + B-G431B-ESC1 + AS5600（美区 BOM **$188** / 中国 **$157**），5010 含 5010 电机（美区 BOM **$136** / 中国 **$94**），整机美区 BOM **$4,312** / 中国 BOM **$3,236**（均低于 $5,000）。
- **通讯与机载计算**：Intel N95 mini PC + 4 路 USB-CAN 2.0（每路 1 Mbps，驱动与 BNO085 IMU 刷新 **250 Hz**）+ 6S 4000 mAh LiPo（约 30 min 续航）；PPO 双足 MLP 策略机载推理 **25 Hz**，行走实测仅消耗执行器极限力矩的 **30%**。
- **台架实测指标（§IV）**：摆线减速箱机械效率 **~90%**；6512 PLA 传输刚度实测 **319.49 N·m/rad**（对比文献 PA-CF 打印摆线 1468 N·m/rad）；60 h 摆锤耐久（±45°–90°, 0.5 Hz）效率回升稳定，背隙缓慢上升但可控；2 台不同打印机 6 只 6512 一致性良好，力矩跟踪误差在 **±0.5 N·m** 内，新打印最大背隙 **0.0229 rad**（标准差 0.0042 rad）；5-DoF 机械臂 SteamVR 追踪 100 次重复定位标准差 **3.433 mm**。
- **已记录局限（§VI）**：论文明确指出对 3D 打印塑料件长时热效应仍缺乏系统研究。
- **知识归纳**：[berkeley-humanoid-lite](../../wiki/entities/berkeley-humanoid-lite.md).

## 对 wiki 的映射

- [berkeley-humanoid-lite](../../wiki/entities/berkeley-humanoid-lite.md)
- 分类父节点：[paper-notebook-category-12-hardware-design](../../wiki/overview/paper-notebook-category-12-hardware-design.md)

## 参考来源（原始）

- [Robot Learning Paper Notebooks · PROGRESS.md](https://github.com/ImChong/Robot_Learning_Paper_Notebooks/blob/main/papers/PROGRESS.md)
- 论文：<https://arxiv.org/abs/2504.17249>
