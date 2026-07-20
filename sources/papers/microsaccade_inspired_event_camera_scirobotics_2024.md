# Microsaccade-inspired Event Camera for Robotics（Science Robotics, 2024）

> 来源归档（ingest）

- **标题：** Microsaccade-inspired event camera for robotics
- **类型：** paper / event camera / neuromorphic vision / hardware-software co-design / robotics perception
- **期刊：** Science Robotics, 2024（Volume 9, Issue 90, eadj8124）
- **DOI：** <https://doi.org/10.1126/scirobotics.adj8124>
- **arXiv：** <https://arxiv.org/abs/2405.17769>
- **项目页：** <https://bottle101.github.io/AMI-EV/>（详见 [`sources/sites/ami-ev-bottle101-github-io.md`](../sites/ami-ev-bottle101-github-io.md)）
- **作者：** Botao He*†、Ze Wang、Yuan Zhou、Jingxi Chen、Chahat Deep Singh、Haojia Li、Yuman Gao、Shaojie Shen、Kaiwei Wang、Yanjun Cao、Chao Xu、Yiannis Aloimonos、Fei Gao*、Cornelia Fermüller*（† 第一作者；* 通讯作者）
- **机构：** 浙江大学（高飞、徐超课题组，ZJU），马里兰大学帕克分校（UMD，Cornelia Fermüller、Yiannis Aloimonos、Botao He），香港科技大学（HKUST，Shaojie Shen）
- **入库日期：** 2026-07-20
- **一句话说明：** 受人眼微扫视（microsaccade）启发，在事件相机前加装旋转楔形棱镜主动偏转光路，使静态场景也能持续产生事件，再通过几何光学补偿算法还原稳定纹理，构建 **AMI-EV** 系统，在强挑战光照与静止场景下同时超越普通相机和传统事件相机。

## 相关资料（策展）

| 类型 | 链接 | 说明 |
|------|------|------|
| 项目页 | [bottle101.github.io/AMI-EV](https://bottle101.github.io/AMI-EV/) | 演示视频、开源代码与仿真工具 |
| arXiv 全文 | [arXiv:2405.17769](https://arxiv.org/abs/2405.17769) | 预印本，含完整方法与实验 |
| 事件相机视角对照 | [KEMO（事件驱动关键帧记忆 VLA）](../../wiki/entities/paper-kemo-event-driven-keyframe-memory-vla.md) | 事件相机在高层决策中的应用对照 |
| 对应 sites 归档 | [AMI-EV 项目页归档](../sites/ami-ev-bottle101-github-io.md) | 开源核查详情 |

## 摘要级要点

- **核心问题：** 事件相机只对 **亮度变化** 响应，若相机静止、场景也静止，则不产生任何事件——**平行于运动方向的边缘会丢失**；这是传感器物理内禀缺陷，纯软件算法难以根治。
- **生物启发解法：** 人眼在注视时会进行无意识的 **微扫视（microsaccade）**——眼球极小幅高速抖动，使视网膜上的感光细胞持续受刺激，防止感知适应（perceptual fading）。本文借鉴这一机制，用 **机械方式主动制造"事件相机的微扫视"**。
- **AMI-EV 系统组成：**
  1. **旋转楔形棱镜（Rotating Wedge Prism）**：装于事件相机镜头前，持续旋转偏转光路，使光在感光面上的投影发生周期性偏移 → 即使场景静止，边缘也持续产生事件。
  2. **几何光学补偿算法**：棱镜旋转引入的额外运动由几何光学公式精确建模 → 算法补偿后恢复稳定纹理外观，输出等效于无棱镜的事件流。
  3. **即插即用（Plug-and-play）**：补偿后的事件流与标准事件相机接口兼容，现有事件驱动算法无需修改即可直接使用。
- **开源内容（已发布）：**
  - 硬件设计（旋转棱镜模块 CAD/电路）
  - 软件工具链（AMI 生成、标定、补偿算法）
  - 仿真平台（基于 WorldGen 3D 场景渲染）
  - 数据集转换工具（Translator，将现有公开事件数据集转换为 AMI-EV 格式）
- **性能评测（三种场景）：**
  - **(i) 结构化环境**：AMI-EV 和普通灰度相机均优于标准事件相机（纹理稳定）
  - **(ii) 非结构化环境**：灰度相机最优，AMI-EV 其次，标准事件相机最差
  - **(iii) 极端光照（强逆光/HDR）**：**仅 AMI-EV 可靠工作**；灰度相机过曝，标准事件相机纹理丢失
- **高层任务验证：** 在低层（特征检测/跟踪）和高层（人体检测+姿态估计）两类任务上验证 AMI-EV 系统优势。

## 核心摘录（面向 wiki 编译）

### 1) AMI-EV 硬件模块

| 组件 | 规格 | 作用 |
|------|------|------|
| 旋转楔形棱镜 | 直流电机 + 减速齿轮 + 绝对位置编码器 | 持续旋转产生光路偏转；57 g（含 ESC）；0.11 N·m@100 rpm |
| 方向传感 | 光电传感器感知棱镜绝对朝向 | 补偿算法所需角度参考 |
| 事件相机 | 标准商用事件相机（具体型号不限） | 棱镜与相机解耦，可适配不同型号 |
| 补偿算法 | 几何光学旋转投影模型 + 运动补偿 | 消除额外旋转运动，输出稳定事件流 |

### 2) 与传统事件相机的核心差异

| 维度 | 标准事件相机 | AMI-EV |
|------|-------------|--------|
| 静止场景 | 无事件 | **持续产生事件** |
| 平行边缘丢失 | 常见 | **基本消除** |
| HDR 能力 | 保留 | **保留**（传感器本身不变） |
| 时间分辨率 | 微秒级 | **保留**（传感器本身不变） |
| 与现有算法兼容 | 原生 | **补偿后兼容**（即插即用） |

### 3) 仿真与数据集翻译工具

- **仿真平台**：基于 WorldGen，可生成指定场景纹理、相机运动与镜头参数的合成 AMI-EV 数据；用于不需要物理硬件的算法开发与测试。
- **Translator**：将 Neuromorphic-Caltech 101 与 Multi Vehicle Stereo Event Camera 等公开数据集转换为 AMI-EV 格式，降低数据壁垒。
- **对 wiki 的映射：** 开源状态与工具链详见 `工程实践` 节。

## 源码开放核查（步骤 2.5）

| 类别 | 状态 | 说明 |
|------|------|------|
| 硬件设计（CAD/电路） | **已开源** | 项目页 + 论文附录 Reference [34] |
| 软件（AMI 生成/标定/补偿） | **已开源** | 同上，已随仿真平台一起发布 |
| 仿真平台 | **已开源** | 基于 WorldGen；代码在项目页可访问 |
| Translator 工具 | **已开源** | 支持 Caltech101 与 MVSEC 数据集转换 |
| 完整推理/训练代码 | **部分开放** | 基础工具链开源；特定任务模型可能需自行训练 |

## 对 wiki 的映射

- 主沉淀：**[`wiki/entities/paper-microsaccade-inspired-event-camera.md`](../../wiki/entities/paper-microsaccade-inspired-event-camera.md)**
- 交叉：**[`wiki/entities/paper-kemo-event-driven-keyframe-memory-vla.md`](../../wiki/entities/paper-kemo-event-driven-keyframe-memory-vla.md)**（事件相机在高层决策中的应用）
- 基础概念：**[`wiki/concepts/sim2real.md`](../../wiki/concepts/sim2real.md)**（仿真到真实的数据迁移，与 Translator 工具相关）
