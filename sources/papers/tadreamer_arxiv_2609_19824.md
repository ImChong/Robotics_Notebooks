# TADreamer: Zero-Shot Language-Guided 3D Navigation for Terrestrial-Aerial Bimodal Robots via Video Imagination（arXiv:2609.19824）

> 来源归档（ingest · 全文消化）

- **标题：** TADreamer: Zero-Shot Language-Guided 3D Navigation for Terrestrial-Aerial Bimodal Robots via Video Imagination
- **类型：** paper / language-guided navigation / TABR / video imagination / zero-shot
- **arXiv abs：** <https://arxiv.org/abs/2609.19824>
- **arXiv HTML：** <https://arxiv.org/html/2609.19824v1>
- **PDF：** <https://arxiv.org/pdf/2609.19824>
- **作者：** Xiangyu Li*、Tiancheng Lai*、Xijie Huang、Ruitian Pang、Siqi Shen、Juncheng Chen、Zaisheng Pan、Chao Xu、Fei Gao、Yanjun Cao（通讯）
- **机构：** 浙江大学（ZJU）工业控制技术国家重点实验室；湖州浙江大学研究院；浙大控制学院（Zaisheng Pan）
- **硬件：** 陆空双模机器人（类 [TriphiBot](https://arxiv.org/abs/2602.01385)）；Odin1（RGB + ToF 点云 + 位姿）；机载 Jetson Orin NX + PX4
- **模型/API：** ChatGPT Sol-5.6（VLM）；Wan2.7 I2V；Depth Anything 3（DA3，RTX 4080）；Zhang et al. TABR 轨迹生成器 [5]
- **代码 / 项目页：** 截至 2026-09-29 **arXiv 与 PDF 未列 GitHub 或项目页**
- **入库日期：** 2026-09-29
- **一句话说明：** **零样本** TABR 语言导航：VLM 写 prompt → Wan 生成第一人称导航视频 → VLM 选片/纠错 → DA3 重建轨迹 → **两阶段点云标定**（FoV 初值 + 各向异性 scaling ICP）→ 模式感知规划；七场景真机；相对 NavDreamer **MADE −87.7% / MARDE −86.3%**。

## 核心摘录（面向 wiki 编译）

### 1) 管线（Fig. 2）

1. **VLM 感知与 prompt：** 观测 clip \(\mathcal{I}^{obs}\) + 指令 \(\ell\) → \(P_{nav}=(d_{route}, d_{mode}, d_{stop})\)。
2. **视频生成：** Wan2.7，每轮 \(n=5\) 候选；VLM 评估 + 选 \(V_{best}\) 或更新 \(P_{fix}\) 再生（\(R_{max}\) 有界）。
3. **模式标注：** VLM 从视频时间轴得 \(\tau^{fly}_k, \tau^{land}_k\)，给采样路点贴 terrestrial / aerial 标签。
4. **几何：** DA3 得相机轨迹与 imagined 点云；与 **实测 ToF 点云** 两阶段配准得 metric waypoints。
5. **执行：** Zhang et al. [5] 模式感知 TABR 规划 + 低层跟踪。

### 2) 两阶段标定（§III-C）

- **Stage 1：** FoV 约束水平轴比 \(\kappa=s_{0y}/s_{0x}\)，scale-only trimmed ICP → \(S_0^\star\)。
- **Stage 2：** 联合 \(S_1, R, t\) 最小化对应点距离；合成 \(S^\star=S_1^\star S_0^\star\) 校准路点。

### 3) 真机（§IV-A · Table I–II）

- **七场景：** Slope、Grassland、KFC、Notice、Red Box 1/2、Square。
- **可用视频：** 5 候选/轮；**两轮内** 七场景均得到可用片；最终选中视频 **Mode Success 全对**（人工参照）。
- **深度（均值）：** Ours MADE **0.490 m** / MARDE **40.34%** vs NavDreamer **3.992 m / 293.78%** vs DA3 **2.446 m / 164.67%**。

### 4) 开源核查（2026-09-29）

- PDF/HTML **无 Code availability 链接** → wiki 标 **待发布/未列仓库**。

## 对 wiki 的映射

- [`wiki/entities/paper-tadreamer.md`](../../wiki/entities/paper-tadreamer.md)
- 交叉：[视觉–语言导航](../../wiki/tasks/vision-language-navigation.md)、[生成式世界模型](../../wiki/methods/generative-world-models.md)
