# ObjectWorld（arXiv:2611.11085）

> 来源归档（paper）

- **标题：** ObjectWorld: Scaling Object-Centric World Models with A Large-Scale 3D Full-Object Flow Dataset
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2611.11085>
- **PDF：** <https://arxiv.org/pdf/2611.11085v1>（v1，2026-11-08，cs.RO；共 12 页，含附录 A–D）
- **项目页：** <https://nus-lins-lab.github.io/ObjectWorld/>
- **作者与机构：** Haonan Chen、Lin Shao（通讯）等；新加坡国立大学计算学院、RoboScience、NUS 广州研究转化与创新研究院
- **代码：** 文章未给；项目页静态 HTML 只有「Paper (PDF)」与演示视频链接，未见代码 / 数据集下载入口（未核实是否另有发布）
- **入库日期：** 2026-11-11
- **一句话说明：** 提出「3D 全物体流」（对完整物体表面含遮挡部分给出稠密 3D 轨迹），构建 1210.1 小时数据集（454.5 h 仿真 + 755.6 h 自动标注真实视频），训练图像 + 语言条件的流预测世界模型，并用轻量优化把流转成末端轨迹。

## 开源状态

- 项目页（2026-11-11 静态读取）：无 GitHub / Hugging Face 链接；有 28 个可交互示例（12 真实 + 16 仿真）。论文正文未写代码或数据集发布承诺。
- 结论：**未见开源入口**；数据集是否发布、何时发布均未核实。

## 核对范围

- **核对范围：** 读了 arXiv v1 PDF 全文（摘要、第 I–V 节、参考文献列表、附录 A 标注管线、附录 B 数据集细节、附录 C 模型与训练细节、附录 D 组件消融），表 I–X 逐表核对，图 4（混淆矩阵）、图 6（数据缩放格点）、表 IV / VII 以渲染图核对数值；图 1、3、5、7 只看了图注与示意。未读：被引文献原文；未运行任何代码；项目页只读静态 HTML 文字，未播放视频、未操作 3D 交互浏览器；PDF 之外无补充材料可读。
- **通读：** 已完成

## 核心摘录

1. **表示（第 III-A 节）：** 给定初始 RGB-D 观测与语言指令，预测 N=512 个物体表面点、T=48 步的 3D 轨迹；以相对初始表面的位移表示（第 III-C 节式 1）。
2. **数据（第 III-B 节、附录 B，表 VIII/IX）：** 总计 1,210.1 小时（16 FPS 标准化），488,923 条样本；仿真 454.5 h（198,947 条轨迹，251 个任务，7 个数据集，MetaFold 提供形变物体），真实 755.6 h（AgiBot World 718.6 h、Sth-V2 34.1 h、EPIC-KITCHENS 2.9 h；289,976 个片段）。Golden 子集（重投影 IoU ≥ 0.70）120,269 条 / 288.7 h，Silver（≥ 0.50）即全部 755.6 h。主模型用全部仿真 + Golden 训练。
3. **标注管线（第 III-B 节、附录 A）：** Gemini 2.5 Pro 选目标物与切片 → Track4World（DA3 骨干）估相机与点云 → SeC 传播掩码 → SAM 3D 重建整物体网格 → Any6D 初始位姿 + FoundationPose 跟踪 → 在网格上均匀采 512 点并按位姿序列变换到首帧相机系。真实数据的轨迹来自刚体位姿跟踪（推断：真实视频里的流是刚体运动，形变数据仅来自仿真网格顶点，见附录 B-C）。
4. **标注验证（第 IV-B 节，表 II / III，图 4）：** 在 99 个 RoboTwin 3 秒片段上，成功标注 97 个（98.0%）；整物体重建 CD/D 2.61%，隐藏表面误差 7.28%D，最终 EPE 3.71 cm（26.11%D）。与 Track4World 部分流相比 EPE/D 37.22%→30.74%，整体覆盖 28.4%→45.8%，隐藏面覆盖 9.8%→26.6%。200 个专家标注真实片段上 Golden 门 precision 93.3%（recall 69.1%），Silver 门 recall 96.3%（precision 65.0%）。
5. **模型（第 III-C 节、附录 C）：** 冻结 Qwen2.5-VL-7B-Instruct 提取点对齐语义特征，PTv3 联合编码物体与场景点云，x1 预测的流生成（线性加噪路径，MSE 损失），物体 token 自注意力后对场景 + 语言 token 交叉注意力。3 个 epoch、8×A100、全局 batch 64；评估 1 步采样，机器人执行 5 步（约 1.08 s / 条轨迹，单 A100）。
6. **预测结果（第 IV-C 节，表 IV）：** 全帧 EPE 6.22 cm / Acc@10 79.3%，末帧 EPE 11.17 cm / Acc@10 60.1%；对比 ObjectForesight 10.24 / 21.18、TraceGen 12.97 / 26.43、im2Flow2Act 12.58 / 20.88（其为横向 2D 流误差，口径不同）。物体点加 5 cm 高斯噪声，末帧 EPE 11.17→11.61 cm。
7. **数据缩放（第 IV-D 节，图 6，末帧 EPE cm）：** 仿真 0/25/100% × 真实 0/25/100%：仿真 0% 行为 —/19.80/18.81；25% 行 17.10/16.79/16.80；100% 行 14.14/13.04/13.01。任务语言消融（表 V）：占位 caption 13.76 → 任务描述 13.01。
8. **执行（第 III-D、IV-E/F 节，表 VI / VII）：** 抓取模块给初始位姿与接触区，局部接触块 Huber 鲁棒 Kabsch（IRLS）对齐得末端轨迹。仿真（ManiSkill PushCube / PlaceSphere，LIBERO Insert / Stack-Arrange）Ours 10/10、10/10、9/10、10/10，对 TraceGen、ObjectForesight 的 4–7/10；仿真「成功」按与真值轨迹的覆盖准则判定（后抓取阶段）。真实（平行夹爪 Banana / Water / Flashlight，灵巧手 Cabbage，各 5 次）：Ours 4/5、4/5、5/5、5/5，对 Dream2Flow 2/5、3/5、5/5、4/5。
9. **附录 D（表 X）：** 去掉 VLM 条件：末帧 EPE 12.69→16.90 cm；物体与场景分开编码（Separate PTv3）：13.46 cm。

## 对 wiki 的映射

- [paper-objectworld](../../wiki/entities/paper-objectworld.md)
