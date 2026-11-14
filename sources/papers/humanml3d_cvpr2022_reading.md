# Generating Diverse and Natural 3D Human Motions from Text（HumanML3D 原文通读归档）

- **标题：** Generating Diverse and Natural 3D Human Motions from Text（CVPR 2022, pp. 5152–5161）
- **作者与机构：** Chuan Guo, Shihao Zou, Xinxin Zuo, Sen Wang, Li Cheng, Wei Ji, Xingyu Li（University of Alberta）
- **论文：** <https://openaccess.thecvf.com/content/CVPR2022/papers/Guo_Generating_Diverse_and_Natural_3D_Human_Motions_From_Text_CVPR_2022_paper.pdf>
- **项目主页：** <https://ericguo5513.github.io/text-to-motion>
- **官方仓库：** <https://github.com/EricGuo5513/HumanML3D>（代码 MIT 许可证；动作数据受上游 AMASS 分发协议约束，不直接随仓库打包发布，需从 AMASS + HumanAct12 本地脚本重建）
- **核对范围：** 通读 `humanml3d_cvpr2022.txt`（CVPR 2022 正文全篇 §1–§6 与参考文献）以及官方仓库 `EricGuo5513/HumanML3D`（`README.md`、`LICENSE`、`paramUtil.py`、`text_process.py`、`common/skeleton.py`、`common/quaternion.py`、`human_body_prior/data/prepare_data.py` 与 `HumanML3D/{all,train,val,test,train_val}.txt` 划分索引）。
- **通读：** 已完成

## 一句话摘要

在 CVPR 2022 同一工作下提出包含 14,616 段 20 fps 人体动作（共 28.59 小时）与 44,970 条人工英语描述（词表 5,371 词）的 **HumanML3D** 数据集，并给出两阶段文本到动作生成框架（`text2length` + 基于 motion snippet code 的三元组 Temporal VAE）及 R-Precision / MultiModal Dist 评测协议。

## 原文核心事实核对

### 1. 数据集构建与规模（§4, Table 1, §5）

- **原始动作来源：** 聚合 **AMASS** 与 **HumanAct12** 两个公开 3D 动捕数据库（涵盖日常步行/跳跃、游泳/空手道/高尔夫等运动、侧手翻等杂技与舞蹈等）。
- **归一化处理：**
  - 全部动作重采样至 **20 fps**；
  - 超过 10 秒的片段随机裁剪至 10 秒（最终片段时长区间 **2–10 s**，平均 **7.1 s**）；训练与测试时进一步裁剪为 **4 帧的整数倍**；
  - 重定向至默认人类骨架模板（22 个 **SMPL** 关节；作为对照，**KIT-ML** 为 21 个关节），并旋正初始朝向至 **Z+ 方向**。
- **文本标注（AMT）：** 在 Amazon Mechanical Turk 雇用母语为英语、历史通过率 > 92% 的标注员，要求每条描述不少于 5 个词，每段动作收集来自不同标注员的 **3 条描述**（经人工后处理过滤异常文本）；描述长度均值 **12 词**、中位数 **10 词**。
- **数据增强与划分：** 通过左右镜像翻转所有动作并替换描述中方向词（如 `left` $\leftrightarrow$ `right`、`clockwise` $\leftrightarrow$ `counterclockwise`）将数据翻倍；按 **0.8 : 0.15 : 0.05** 划分为 `train / test / val`。
- **对比 Table 1：**
  - **HumanML3D：** 14,616 motions / 44,970 texts / 28.59 h / 5,371 vocab；
  - **KIT-ML：** 3,911 motions / 6,278 texts / 10.33 h / 1,623 vocab（§1 引言写作 KIT-ML 3,010 sequences，§2 与 Table 1 给出的完整数字为 3,911 sequences / 6,278 sentences / 10.33h）。

### 2. 动作表示与生成方法（§3, §5）

- **单帧姿态元组 $p$（§5）：** $p = (\dot{r}^a, \dot{r}^x, \dot{r}^z, r^y, j^p, j^v, j^r, c^f)$：
  - $\dot{r}^a \in \mathbb{R}$ 为绕 Y 轴的根角速度；$(\dot{r}^x, \dot{r}^z) \in \mathbb{R}^2$ 为 XZ 平面根线速度；$r^y \in \mathbb{R}$ 为根高度；
  - $j^p, j^v, j^r$ 分别为根空间局部关节位置、速度与 6D 连续旋转表示（参考 Zhou et al., CVPR 2019）；
  - $c^f \in \mathbb{R}^4$ 为依据脚跟与脚尖关节点速度阈值提取的二值足地接触特征。
- **Motion Autoencoder & Snippet Code（§3.1）：** 两层 1D 卷积（`filter=4, stride=2`）将 $T'$ 帧压缩为 $T = T'/4$ 个 **motion snippet code** $\mathbf{c}_s^t$（单码感受野 8 帧，约 0.4–0.5 s）；反卷积解码器 $D$ 额外预测每帧足地接触以抑制滑步，训练损失叠加 L1 重建、稀疏约束与相邻码平滑约束。
- **Text2length 采样（§3.2）：** 基于句子级文本特征经 MLP + Softmax 输出离散长度索引 $\{1, \dots, T_{\max}\}$（$T_{\max}=50$ 对应 200 帧即 10 s）的多项分布，以交叉熵训练。
- **Text Encoder + Temporal VAE（§3.3）：**
  - Bi-GRU 文本编码器融合词向量、**POS 词性标签**以及人工整理的 4 类运动词典（`direction`、`body part`、`object`、`action`）标签；
  - 由先验网络 $F_\psi$、后验网络 $F_\phi$、生成器 $F_\theta$ 组成时间循环 VAE，结合 **Local Word Attention** $F_{\text{att}}$ 与 **Time-to-Arrival 位置编码** $\mathrm{PE}_{T-t}$；采用课程学习（逐步拉长生成片段长度 $T_{\text{cur}} \to T_{\max}$）与计划采样（teacher forcing 概率 $p_{tf}$）。

### 3. 定量实验与局限（Table 2, Table 4, §5.1.4）

- **对比基线（Table 2, HumanML3D 测试集，20 次重复 95% 置信区间）：**
  - Real motions：R-Precision Top-1/2/3 = `0.511 / 0.703 / 0.797`，FID = `0.002`，MultiModal Dist = `2.974`，Diversity = `9.503`；
  - `Ours w/ real length`：R-Precision = `0.457 / 0.639 / 0.740`，FID = `1.067`，MultiModal Dist = `3.340`，Diversity = `9.188`，MultiModality = `2.090`；
  - `Ours`（含 `text2length` 采样）：R-Precision = `0.455 / 0.636 / 0.736`，FID = `1.087`，MultiModal Dist = `3.347`，Diversity = `9.175`，MultiModality = `2.219`。
- **消融与局限（Table 4 & §5.1.4）：** 去掉 `SnC`（snippet code）或 `Att`（局部词注意力）使 R-Precision 下降超过 6%，序列后期极易偏离文本指令；AMT 主观偏好测试中约 72% 生成结果进入用户偏好 Top-2；受篇幅限制，论文将具体失败案例与进一步局限讨论放在补充材料。

## 官方源码核对结果

对 `EricGuo5513/HumanML3D` 仓库核对如下：
- **开源状态：** 已开源（代码 MIT；受 AMASS 分发政策限制，`new_joints` / `new_joint_vecs` 等动作实体不直接随 GitHub 发布，用户需下载 AMASS 与 HumanAct12 及 SMPL+H/DMPL 模型后本地重建；`KIT-ML` 处理后数据提供 Google Drive 下载链接）。
- **仓库实际文件：**
  - `paramUtil.py`：定义 `t2m_kinematic_chain`（5 条运动链、22 关节 SMPL，以及 `t2m_left_hand_chain` / `t2m_right_hand_chain` 手部扩展链索引）、`t2m_raw_offsets`、`t2m_tgt_skel_id = '000021'` 与 KIT-ML 的 21 关节骨架参数；
  - `text_process.py`：使用 `spacy.load('en_core_web_sm')` 对英文描述做分词、词形还原（`NOUN`/`VERB` 保留 `left` 原形）与 POS 标注，输出 `原始文本#词/POS序列#起止时间` 格式写入 `./texts/*.txt`；
  - `common/skeleton.py` & `common/quaternion.py`：实现 `Skeleton` 类的骨长重采样（`get_offsets_joints`）、基于双髋与双肩向量叉乘确定朝向并旋转至 Z+ 的逆运动学（`inverse_kinematics_np`）及基于 6D 旋转 / 四元数的前向运动学（`forward_kinematics_cont6d`）；
  - `HumanML3D/{all,train,val,test,train_val}.txt`：含镜像样本前缀 `MXXXXXX` 在内的精确样本 ID 划分（`all.txt` 共 **29,228** 行 = 原版与镜像成对序列，`train.txt` **23,384** 行，`test.txt` **4,384** 行，`val.txt` **1,460** 行，严格符合 `0.8 : 0.15 : 0.05`）；
  - `README.md` 列出 4 个预处理 Notebook（`raw_pose_processing.ipynb`、`motion_representation.ipynb`、`cal_mean_variance.ipynb`、`animation.ipynb`），但当前本地克隆快照未包含 `.ipynb` 文件实体，运行前需核对上游仓库发布路径或自行按 `Skeleton` / `paramUtil` 接口组织预处理入口；此外 `README.md` 特别提示 `matplotlib==3.3.4` 可避免动画导出或重采样时的细微数值偏差。

## 知识库映射

- [HumanML3D（BFM 行为数据）](../../wiki/entities/dataset-bfm-humanml3d.md)
