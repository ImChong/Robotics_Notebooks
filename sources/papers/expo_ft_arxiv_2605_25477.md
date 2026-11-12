# EXPO-FT: Sample-Efficient Reinforcement Learning Finetuning for Vision-Language-Action Models（arXiv:2605.25477）

> 来源归档（paper · 原文通读 + 2026-09-29 公众号策展深化）

- **标题：** EXPO-FT: Sample-Efficient Reinforcement Learning Finetuning for Vision-Language-Action Models
- **作者：** Perry Dong*、Kuo-Han Hung*、Tian Gao、Dorsa Sadigh、Chelsea Finn（Stanford University；* 同等贡献）
- **arXiv：** <https://arxiv.org/abs/2605.25477>（v1，2026-05-25，cs.RO）
- **PDF：** <https://arxiv.org/pdf/2605.25477v1>
- **项目页：** <https://pd-perry.github.io/expo-ft/>
- **代码：** <https://github.com/pd-perry/expo-ft>（论文摘要称发布开源代码库；仓库细节见 [expo-ft.md](../repos/expo-ft.md)，正文未给出仓库 URL）
- **备注：** CoRL 2026（会场信息来自清单与既有归档，论文 PDF 首页未标注）；基座 π0.5；算法源自 [EXPO](./expo_arxiv_2507_07986.md)
- **入库日期：** 2026-09-29（原文通读升级：2026-11-12）
- **核对范围：** 通读 arXiv v1 PDF 全部正文 §1–§7（含式 1–5、表 1–2、图 1–4 的图注与坐标轴）与附录 A（训练曲线 / 回合时长）、B（八任务示教量、奖励与复位、随机化）、C（模型结构、表 3–4 超参、基线实现）；图 4 / 5 / 6 / 7 的曲线与照片仅按图注、坐标轴文字和正文结论核对，未逐点读曲线；参考文献只看了与正文论断对应的条目，未打开被引论文；未运行任何代码，未核对 GitHub 仓库与项目页视频。
- **通读：** 已完成
- **一句话说明：** 在 π0.5 上用 EXPO 式 base 提案 + edit + Q 选优（扩展到 action chunk，加人工干预）做在线 RL 微调；八个真机任务各 30 次评测全部成功，平均 19.1 min 在线数据。

## 核心摘录（均出自论文）

1. **问题（§1、§4.1）：** 预训练 VLA 零样本成功率不足以部署；既有 RL 要么从头训练、不用 VLA 先验，要么微调 VLA 但不够可靠或样本太多。EXPO 原文不支持 action chunk 与人工干预，EXPO-FT 补上这两点（§4.1）。
2. **方法（§3、§4.2）：** 基座 π_VLA 用监督损失训练，轻量 edit policy 最大化 Q（式 1），edit 限制在 [-β, β]，不对 VLA 反传；推理与 TD 备份用 OTF 策略在 base 与 edited 候选中取 Q 最大者（式 2–3）。chunk 版本见式 4–5（H 步预测、执行 C ≤ H 步）。基座 VLA 仍用原训练目标更新。
3. **人工干预（§4.2、§4.3）：** 可在 chunk 内任意连续时间步用 SpaceMouse 覆盖动作，执行后写入回放缓冲；干预率早期高、随训练降到 0。
4. **流程（§4.2）：** 零样本不行则收集示教并模仿微调到成功率约 40% 以上；示教直接初始化回放缓冲；之后在线 RL；更新可按步 / 按回合 / 按大批回合。
5. **系统（§4.3）：** learner（VLA 训练与推理、回放缓冲）与 actor（真机交互）解耦，可同步或异步；GPU ≤ 2 时用同步。critic 用独立轻量 ResNet-50，不共享 VLA 编码器。奖励为任务专用的规则式稀疏二值检测器（精度 >95%，§5.1）。
6. **设置（§5.1、附录 C）：** 末端笛卡尔速度控制 10 Hz；侧视 + 腕部 RGB 224×224 + 本体状态；两块 H200；每任务 8k–20k 环境步；π0.5 来自任务专用 LoRA SFT 检查点；10 个 Q 网络的 RedQ 式集成，目标取随机 2 个的最小；每次决策 8 个 base chunk + 8 个 edit chunk 共 16 候选，确定性 top-Q；edit 缩放 0.05 / 0.1 / 0.2（表 4 实际用 0.05 或 0.2）；UTD 20、γ=0.99、batch 64、Adam 3e-4。
7. **结果（表 1、表 2）：** 八个任务（Egg Flip、String Light Routing 的 Route I / Route II / Insert、Candy Scoop、Cube Pick、Flower Insert、Pool Shot）EXPO-FT 均 30/30；在线数据 14–35 min，平均 19.1 min。SFT 平均 20.5/30，HG-DAgger 平均 22.1/30（八任务）。HIL-SERL 与 DSRL 只在 Egg Flip / Cube Pick / Pool Shot / Flower Insert 四个任务上比：SFT 18.8、HG-DAgger 20.5、DSRL 19、HIL-SERL 5.5、HIL-SERL(M) 20（两个任务）、EXPO-FT 30（表 1）。
8. **示教量（附录 B.1）：** Egg Flip 25、Route I 40、Route II 30、Insert 25、Candy Scoop 20、Cube Pick 10、Flower Insert 25、Pool Shot 30 条。
9. **作者自述局限（§6）：** 需要人工在回合间复位；VLA 参数量大，高频训练与推理算力仍吃紧。

## 公众号 / 博客（二手，未逐句核对）

- [wechat_embodied_heart_expo_universal_post_training_2026-09-29.md](../blogs/wechat_embodied_heart_expo_universal_post_training_2026-09-29.md)
- [pd_perry_universal_post_training_robotics_2026-09.md](../blogs/pd_perry_universal_post_training_robotics_2026-09.md)
- [`wechat_duomo_vla_weekly_trends_2026-08-17_part1.md`](../blogs/wechat_duomo_vla_weekly_trends_2026-08-17_part1.md)

## 对 wiki 的映射

- [`wiki/entities/paper-expo-ft.md`](../../wiki/entities/paper-expo-ft.md)
- [`wiki/entities/paper-expo.md`](../../wiki/entities/paper-expo.md)
