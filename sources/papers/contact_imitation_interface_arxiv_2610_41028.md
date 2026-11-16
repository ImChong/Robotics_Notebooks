# CIMI — Contact Imitation Interface: Transferring Dexterous Skills by Showing How Contact Unfolds

> 原始论文全文深读与复现边界归档（arXiv:2610.41028v1，2026-10-29）。

- **类型：** paper
- **英文标题：** Contact Imitation Interface: Transferring Dexterous Skills by Showing How Contact Unfolds
- **中文标题：** 接触模仿接口：通过接触演化过程迁移灵巧操作技能
- **作者：** Zhixuan Xu、Yichen Li、Bingchen Wu、Rui Wang、Yiwen Hou、Zixuan Liu、Tianyu Qiu、Lin Shao
- **机构：** 新加坡国立大学（National University of Singapore）、RoboScience、华南理工大学（South China University of Technology）
- **arXiv：** https://arxiv.org/abs/2610.41028
- **原文 HTML：** https://arxiv.org/html/2610.41028v1
- **PDF：** https://arxiv.org/pdf/2610.41028
- **补充材料：** https://nus-lins-lab.github.io/CIMIWeb/supplement.pdf
- **项目主页：** https://nus-lins-lab.github.io/CIMIWeb/
- **代码：** https://github.com/ariszxxu/CIMI
- **入库日期：** 2026-11-03；2026-11-09 原文深读更新
- **核对范围：** 已通读 arXiv 官方 HTML v1 的论文正文 §I–V（包含相关工作、D1–D4/L1–L3 机制、表 I–IV、Q1–Q6 实验、结论与引用）及同页补充材料 I–II（硬件 BOM、AprilCube/DeepTag 定位、重定向优化、相机外参、数据质量、规划器/控制器与训练参数、部署时序及推理延迟）；并核对官方项目页、真实代码仓库和文档的开源边界。未在本环境逐页复核 16 MB 主 PDF，采用作者上传的官方同版本 HTML 全文；没有独立训练或真机复现。
- **通读：** 已完成
- **对 wiki 的映射：** [CIMI 主实体](../../wiki/entities/paper-contact-imitation-interface.md)

## 原文问题与核心贡献

1. **接触对齐的示教硬件（D1–D2）**：人手与 LEAP/Wuji 机器人手使用相同名义接触面几何与指尖 RGB 接触相机位置；指根相机以切向视角观察手指—物体局部区域。三路接触视角、四路腕部/虎口场景视角、两路定位视角共九路视频以 **10 Hz** 同步采集。AprilCube 标记 + DeepTag 密集角点 + 多标签 PnP 重建接触面的 SE(3) 轨迹。
2. **接触优先的重定向（D3）**：针对人手与机器人手结构不同，联合优化机器人手腕位姿和活动关节，以各接触面的四角点位置误差与时序平滑项为目标；不是简单把人手关节角复制给机器人。优化在关节界限下通过 PyRoki 进行。
3. **场景视角变换（D4）**：指尖接触相机因为几何共享，保留原始人手示教画面；腕部/虎口场景相机因手腕重定向导致视角变化，需用双目深度、点云重投影、机器人几何覆盖生成机器人视角 RGB。**深度用于离线场景重投影及在线场景预处理，最终策略看到的是 RGB，不是深度或三维点云。**
4. **接触面动作表征与分层策略（L1–L3）**：以当前接触面为参考，预测各指接触面的相对 SE(3) 轨迹（3 平移 + 6 维旋转，单步每指 9 维）。冻结 C-RADIOv3-B 视觉编码器，与拇指相对指间位姿结合，通过六残差块的条件 Flow Matching 规划器预测 **16 步**；独立八残差块 MLP 控制器将接触面目标 + 当前关节状态转为手腕与手指关节命令。控制器经过合成动作扰动、前向运动学接触四角点损失训练；训练规划器时控制器冻结。
5. **闭环推理（§III-B、补充 II-C）**：16 步轨迹采样 10 Hz、标称覆盖 1.6 秒，每次只执行 8 步再更新观测并重规划。**10 Hz 是动作轨迹采样率，不能误读为固定的视觉重规划频率。**

## 关键实验：原文 Table II，成功率（SR，%）

每个方法 × 手 × 任务运行 **20 次**；六个手—任务对各取最终阶段 SR 的等权平均：

| 方法 / 视觉输入 | LEAP 方块 Lift | LEAP 硬币 Lift | LEAP 信封 Letter Lift | LEAP 灯泡 Light | Wuji 方块 Lift | Wuji 硬币 Lift | 均值 |
|---|---:|---:|---:|---:|---:|---:|---:|
| WristJoint / CTX | 0 | 0 | 0 | 5 | 0 | 0 | 0.8 |
| ContactSurface / CTX | 0 | 0 | 0 | 10 | 0 | 0 | 1.7 |
| WristJoint / CTX+CC | 20 | 35 | 15 | 70 | 15 | 30 | 30.8 |
| ContactSurface / CTX+CC | **75** | **85** | **80** | **90** | **70** | **85** | **80.8** |

说明：CTX 是选定场景 RGB；CC 是对应的局部接触相机 RGB。**1.7→80.8%** 是同 ContactSurface 策略增加 CC 的消融；**30.8→80.8%** 是固定 CTX+CC 时由直接 WristJoint 变为 ContactSurface 分层规划的消融；两者不能混成单一因果对比。

- **原文 Table III（重定向）**：接触角点优先匹配 vs 场景对齐角点匹配，硬币角点均值误差 **14.67→0.51 mm**，硬币 Lift **0→85%**；灯泡角点误差 **5.72→1.09 mm**，Light **0→90%**。每种变体每个任务 20 次。
- **原文 Fig.6（采集效率）**：15 分钟同一操作者/同标准成功示教，CIMI 达裸手成功示教吞吐的硬币 **70%**、灯泡 **71%**；灯泡比测试的遥操作设置 **2.3×**。测试条件与方法特定，不是一般遥操作速度定律。
- **原文 Table IV（LEAP coin 与 bulb）**：去掉机器人几何叠加后最终均值 **62.5%**；既不重投影也不叠加降到 **30.0%**；移除规划器 proprioception **42.5%**；以关节值替换拇指相对位姿 **52.5%**；禁用规划器视觉增强 **0%**；任务专属控制器且不做运动增强 **52.5%**；完整方案 **87.5%**。这里平均仅覆盖硬币 Lift 与灯泡 Light，不是上述六任务总均值。

## 硬件、效率、数据与复现边界（补充 I–II）

- **采集硬件 BOM**：三枚 HBVCAM-2307-FPC82 V11 接触相机（约 8.6×8.6×5.4 mm、120° FOV）、两套 Orbbec Gemini 305 双目、两套 DCXG500 定位相机，以及 3D 打印、硅胶套和摩擦胶带；论文估价 **$778.95**，不含人工、打印机折旧、线缆、主机。
- **数据**：LEAP 方块 146、硬币 187、Letter 183、灯泡 103 条训练示教；Wuji 方块和硬币直接复用相应的**人类原始示教**，但需要单独转换机器人动作标签及视角。
- **处理吞吐**：作者在 RTX 4090 D 处理 10 个硬币示教 / 557 帧，视觉跟踪 57 s + 重定向 43 s + 渲染 72 s + RADIO 缓存 52 s，串行总 **224 s**。
- **规划器训练**：batch 256、最多 63k optimizer updates、AdamW 1e-4；controller batch 512、最多 20k updates，先 1000 步动作回归预热。
- **推理速度**：RTX 4090 D、两接触 + 一场景 RGB、16 目标、3 flow step，准备好的 RGB→命令中位 **7.47 ms**（p95 7.91），缓存特征→命令中位 **2.66 ms**，控制器本身 **0.84 ms**；**不含相机采集、立体匹配、栅格化、通信**。
- **力感知边界**：不是独立的力/压力传感器，也不显式预测接触力；靠近接触面的视觉线索、被动胶带顺应性与相对运动反馈闭环完成任务。灯泡需要三指多次旋转、释放、重新接触。
- **泛化边界**：评测仅 LEAP/Wuji 两种手与六个手—任务对，尚非双手/移动操作普适验证；现场相机布局、左右手映射、标定和控制器合同需保持一致。
- **资源开放程度（2026-11-09 复核）**：公开了处理/训练/部署源码与硬件说明；**原始四套人类示教集、六套训练 Zarr、已训练策略和低层 decoder 仍未提供公开下载路径**。参见官方 [RESOURCES.md](https://github.com/ariszxxu/CIMI/blob/main/docs/RESOURCES.md)。源码可读并不等于外部从零复现论文全部指标。

## 关联与入口

- [CIMI 主实体](../../wiki/entities/paper-contact-imitation-interface.md)
- [项目页与硬件说明核验](../sites/contact-imitation-interface.md)
- [官方可运行源码入口与开放性核验](../repos/contact_imitation_interface.md)
- [arXiv v1 官方 HTML（含补充材料）](https://arxiv.org/html/2610.41028v1)
