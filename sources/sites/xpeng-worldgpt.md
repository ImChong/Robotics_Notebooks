# WorldGPT Project Page | 小鹏 WorldGPT 项目页

- **类型：** site / project-page
- **标题：** WorldGPT: One World Action Model for Everything That Moves
- **机构：** 小鹏（XPeng），XPeng PWM Team
- **作者：** Zhenli Zhang、Yuan Lin、Rui Guo、Yutong Zheng、Junhong Zhou、Hanpeng Liu、Liu Yang、Xu Han、Baolu Li、Jingyu Qian、Zhuangzhuang Ding、Yu Zhang、Xianming Liu
- **项目页：** <https://worldgpt-1.github.io/>
- **官方发布：** [小鹏中文发布](https://www.xiaopeng.com/tech_research/5624.html) · [XPENG 英文发布](https://www.xpeng.com/pressroom/news/01a20bcf44b3a1e838d68a02a2110153)（2026-11-24）
- **版本：** 2026-11-25 读取的公开项目页；页面评测快照日期 2026-11-12。
- **入库日期：** 2026-11-25
- **核对范围：** 通读官方项目页正文，核对跨本体演示、云端教师、能力、可控未来、作者署名和完整榜单注释；检查页面的公开展示数据 `content.js` 与小鹏发布稿。此记录覆盖项目页，不是待发布技术报告或视频复现实验。
- **通读：** 已完成
- **开放状态：** 项目页标注技术报告待发布，未列模型训练、推理或部署实现、权重与自有训练数据下载。网页展示脚本及演示片段来源不等于模型开源。
- **对应节点：** [小鹏 WorldGPT](../../wiki/entities/project-xpeng-worldgpt.md)

## 原始资料摘记

- 一套权重面向汽车、船、无人机、单臂、双臂及人形。约一秒历史摄像头视频与指令作为条件，展示未来视频、本体动作、感知图和语言解释。动作举例包括车辆或飞行/航行轨迹与操作端运动。
- 项目方以视觉作为跨本体共训的公共接口，将云端教师蒸馏为分本体端侧学生作为部署设计。未公布跨本体贡献的受控消融、学生实现与部署指标。
- 演示清单有 32 条；汽车使用内部素材，其他示例包括 CANOE、UAV-Flow、BridgeData V2、RoboMIND 2.0、Humanoid Everyday。页面将生成未来与真实录像并排，而非 32 场模型控制实机试验。
- 项目方截至 11 月 12 日自报在驾驶、操作、视觉理解、视频生成共 12 项第一，包括并列。各评测有独立协议；其比较未统一训练数据。WOD-E2E 测试集 RFS 8.3132；MT50 95.2 是四个难度组等权；LIBERO-Pro 2.0 的 73.3 / 80.1 / 76.5 分别对应静态、运行时、按 167 个单元汇总的综合分。
- LIBERO-Pro 不是 LIBERO-Pro 2.0；原版页面表格的 70.85 使用扰动指令，项目方列明默认执行器指令口径为 65.68。WorldOlympiad 榜单同时给物理子项和总项，0.890 是物理项，0.683 才是总项（与 LingBot-World 并列）。VBench 2.0 和 VBench-I2V 的 WorldGPT 成绩注明为使用提示改写的自评。
- 驾驶改指令演示同时改变完整路线引导与航点；原场景使用地图图像而替换版本只用文字。不能把它当成只改一条文字、其余条件完全相同的受控实验。
- 2024 年同名论文 [WorldGPT: Empowering LLM as Multimodal World Model](https://arxiv.org/abs/2404.18202) 为独立工作，其仓库不是小鹏的模型源码。
