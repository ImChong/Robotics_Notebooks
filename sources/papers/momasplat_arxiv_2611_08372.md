# MOMASPLAT（arXiv:2611.08372）

> 来源归档（paper）

- **标题：** MOMASPLAT: Dynamic Novel-View Demonstration Augmentation for Mobile Manipulation
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2611.08372>
- **PDF：** <https://arxiv.org/pdf/2611.08372v1>
- **项目页：** https://robin-lab.cs.utexas.edu/momasplat/
- **代码：** 论文结论称「将公开发布代码」（Sec. V）；入库时未核查仓库是否已上线
- **作者：** Sateesh Kumar*、Arnav Balaji*、Arpit Bahety、Skand Peri、Georgios Pavlakos†、Roberto Martín-Martín†（*项目负责人，†通讯 / PI）
- **机构：** The University of Texas at Austin（首页脚注；致谢注明在 Robot Interactive Intelligence (RobIn) Lab 完成）
- **版本：** v1，2026-11-05（cs.RO）
- **入库日期：** 2026-11-09
- **核对范围：** 下载 arXiv v1 PDF，核对首页标题与编号一致；pdftotext 通读正文 Sec. I–V、致谢、参考文献、附录 A 至 D（基线细节、方法细节、基座位姿采样、模仿学习与超参表 Table II）全文；图（Fig. 1–5）只读了图注与正文对图的描述，未逐张查看图像，Fig. 4 的分任务曲线数值仅采用正文明文给出的数字；未访问项目页视频，未运行任何代码。
- **通读：** 已完成
- **一句话说明：** 把移动操作示范中的静态场景（3DGS）与被操作物体（前馈 3D 基础模型估位姿 + 独立 3DGS）分开重建，在采样的新底盘位姿下重渲染并按运动学重定向动作，合成含接触后物体运动的新视角示范，用于训练对底盘位姿偏差鲁棒的 Diffusion Policy。

## 核心摘录

1. **问题（Sec. I）：** 视觉运动策略对相机视角敏感；移动操作中因导航不确定、定位漂移等，部署时底盘位姿很少与示范采集位姿相同。已有 NVS 增强假设静态场景，接触后物体运动无法重渲染，只能增强接触前动作。
2. **关键观察（Sec. I、III）：** 常规移动操作数据采集本身就提供多视角：导航轨迹（物体静止）覆盖场景，操作轨迹提供物体中心观测。
3. **方法三块（Sec. III-A/B）：** 静态场景用 COLMAP + WildGaussians 式不确定性加权 3DGS（遮掉物体与机器人），用机载深度中位数比值恢复米制尺度；物体用 Grounded-SAM / SAM 2 分割，FastVGGT 对物体掩码帧估计相机（物体相对）位姿，用导航帧做桥、Umeyama 求 Sim(3) 对齐到世界系，再联合优化位姿与物体 3DGS；机器人用 OmniGibson 按 IK 关节配置渲染；三路按深度排序做逐像素 alpha 合成。
4. **动作重定向（Sec. III-B）：** 用 SE(2) 变换 ΔT = T̃⁻¹·T 把末端位姿、平移增量、旋转增量轴转到新底盘系，夹爪指令不变。
5. **训练（Sec. III-C、附录 D）：** Diffusion Policy，源 : 合成采样比 0.25 : 0.75。
6. **设置（Sec. IV-A、附录）：** TIAGo，头部 RGB-D 相机，5 个真机任务（PickPlaceBottle、OpenDrawer、PickPlaceRemote、CloseLaptop、OpenAirfryer）、3 个场景；源示范 CloseLaptop 75 条、其余各 90 条，分布在 3 个底盘位姿；9 条导航轨迹；每条源示范生成 10 条合成示范，采样区域 1.2 m × 1.2 m；每任务 12 个未见底盘位姿 × 3 次试验。
7. **结果（Sec. IV-B）：** 五任务平均未见位姿成功率 37%，最强基线 One-Thousand Trajectories 17%，Source Only 6%、VISTA 8%、Action Augmentation 6%；原始底盘位姿下各方法相当（3 个原位姿各 15 次试验）；PickPlaceBottle 在 30 cm 处 33%、CloseLaptop 在 40 cm 处 50%，此时所有基线为 0。
8. **源示范数量消融（Sec. IV-B）：** PickPlaceBottle 用 10 / 50 / 90 条源示范，成功率 33% / 42% / 50%（4 个距离 10–20 cm 的测试点）；90 条不增强仅 8%。
9. **生成质量（Sec. IV-B，Table I）：** 仿真 SimPickPlaceBottle，75 条生成示范、2,486 帧；PSNR 全图 21.27、场景 26.40、物体 18.18、机器人 17.43 dB；物体位姿误差（30 条）平移 1.46 ± 0.71 cm、旋转 4.01° ± 4.66°。真机：场景 24.00 dB（110 帧留出），物体 23.63 dB（VGGT 直接注册位姿）/ 22.59 dB（插值位姿）。
10. **局限（Sec. V）：** 假设大体静态场景与刚性可见物体；可变形物体、严重遮挡、弱纹理困难；物体初始位置被遮挡的区域（抬起的瓶子下方、打开的抽屉或炸锅内部）未被建模；依赖开放世界分割；重放源物体轨迹，不建模接触力与动力学，产生非物理示范；视角质量取决于导航数据覆盖。

## 对 wiki 的映射

- [paper-momasplat](../../wiki/entities/paper-momasplat.md)
