# 运动小脑 64 篇长文通读（具身智能研究室）

> 来源归档（原文通读 · 支撑 `wiki/overview/humanoid-motion-cerebellum-technology-map.md`）

- **标题：** 万字长文｜人形机器人的运动小脑会不会成为人形机器人的基础设施？我翻完60多篇人形运控论文后的一个判断
- **作者：** 具身智能研究室（2026-06-17）
- **类型：** blog / survey / humanoid-motion-control
- **原文：** <https://mp.weixin.qq.com/s/Kx9myecE1Z0eGqOapoqQnA>（全文抓取见 [wechat_motion_cerebellum_64_survey_2026-06-18.md](../raw/wechat_motion_cerebellum_64_survey_2026-06-18.md)）
- **一句话说明：** 把 64 篇人形运控论文放进「动作小脑」框架：走路是底座，全身跟踪是身体接口，Loco-Manip 是任务形态；动作小脑夹在 VLA / 世界模型与真实身体之间。
- **核对范围：** 通读抓取全文 918 行：开篇判断、01–07 七节论述（locomotion 底座、跟踪不是播放器、数据炼油链、跟踪规模化、reference 到 prompt、Loco-Manip、工程闭环）、08 节 A–I 九组共 64 条论文说明（题目 / 机构 / 项目页 / 放置理由）与结语。未逐篇阅读这 64 篇被评述的原论文；各论文的方法细节仅采信作者的一句话概括，未独立核对。图片仅看图注。
- **通读：** 已完成

## 核心摘录（均出自原文）

- **总判断：** 走路是底座，全身跟踪是身体接口，Loco-Manip 是任务形态；动作小脑把参考动作、遥操作信号、稀疏目标、任务空间命令乃至上层模糊意图，翻译成全身可执行、可恢复、可迁移的运动。
- **Locomotion（01）：** 底座而非旧技术，但接口粗（速度、方向、路径、局部地形）。
- **跟踪（02）：** 从 DeepMimic 范式、AMP/SMP 运动先验、PHC 到 MaskedMimic；「给完整 reference 是播放器，给稀疏目标还能补全身体才像动作小脑」。
- **数据（03）：** GVHMR/TRAM（视频到世界坐标）、GMR/NMR/OmniRetarget（重定向）、HumanX/HDMI/SUGAR/GenMimic（视频到交互数据）；reference 脚滑、穿地、自碰、根部漂移会污染 tracker。
- **规模化（04）：** SONIC 规模化、HoloMotion/HumanoidGPT 扩数据源、LIMMT 强调质量与覆盖、Any2Track/RGMT 强调扰动鲁棒。
- **接口变宽（05）：** BFM-Zero、MuGen、OMG、MotionWAM 以 goal / reward / prompt / latent 调用身体。
- **Loco-Manip（06）：** CEER、HANDOFF 等回答「上层给全身控制器什么命令」；DoorMan、HOIST、SplitAdapter、HALO 暴露接触、反作用力、负载变化问题；「Loco-Manip 不是动作跟踪的替代品，而是动作跟踪进入真实任务之后的形态」。作者同时指出当前准确度仍差、机械感强（开篇）。
- **工程观（07）：** 动作小脑上接人类动作 / 遥操作 / VLA / 世界模型，下接真机、Sim-to-Real、力反馈、接触与安全；闭环短（仿真训练—上机—失败录像—改 reference/reward/接口/数据—再上机）；脚滑、漂移、穿地、抗扰、换本体重训、关节抖动、开门被反拉等「土问题才是真问题」。
- **论文地图（08）：** A 走路底座 10 篇（01–10）、B 模仿源流 5 篇（11–15）、C 数据入口 9 篇（16–24）、D 全身跟踪 13 篇（25–37，含 ConstrainedMimic / SafeWBC / SafeFall 安全层）、E 可提示 4 篇（38–41）、F 跨本体与遥操作 5 篇（42–46）、G Loco-Manip 接口 5 篇（47–51）、H 真实任务 8 篇（52–59）、I 柔顺与接触 5 篇（60–64）。部分条目机构标「待核对」。
- **结语：** VLA 负责想，世界模型负责预演，人形机器人要真正动起来，中间必须有一层接住动作意图的身体系统。

## 对 wiki 的映射

- [`wiki/overview/humanoid-motion-cerebellum-technology-map.md`](../../wiki/overview/humanoid-motion-cerebellum-technology-map.md)
