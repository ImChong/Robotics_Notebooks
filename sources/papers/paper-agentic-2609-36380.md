# LEGO-Anything: Coding Agents for 3D Scene Reconstruction

- **作者：** Xirui Li, Peng Shi, Mingwen Dong, Sheng Zhang, Zhuoyan Xu, Dongkyu Lee, Shuaichen Chang, Yi Xiang, Lin Pan, Jiarong Jiang
- **机构：** University of Maryland, College Park；AWS（第一作者在 AWS 实习期间完成）
- **版本 / 载体：** arXiv:2609.36380v1 `[cs.CV]`（2026-09-28 提交）；arXiv HTML 全文与项目官网
- **原文：** [arXiv:2609.36380](https://arxiv.org/abs/2609.36380) / [arXiv HTML](https://arxiv.org/html/2609.36380)
- **项目页 / 代码：** [https://lego-anything.com](https://lego-anything.com)；项目页标注 `Code · Coming soon` 与 `Early Access` 入口，截至 2026-11-27 尚未公开官方代码仓库 URL（开源核查状态：**宣称将开源或未列链接**）。注：论文 Appendix I 引用的 `https://github.com/NousResearch/hermes-agent`（pinned revision `aa6f77596b`）为所接入的第三方 Agent Harness，非本论文官方代码仓。
- **核对范围：** 通读 arXiv HTML 全文（Abstract、§1–§7 正文、Table 1–5、Figure 1–9，以及 Appendix A–I、Table 6–9、Figure 10–24 文字、公式与附录实验说明）及项目页 `https://lego-anything.com` 文本快照。
- **通读：** 已完成

## 原文核对笔记

- 对 wiki 的映射：[LEGO-Anything 实体页](../../wiki/entities/paper-agentic-2609-36380.md)

### 1. 问题设定与框架结构（§1, §3）
- 核心命题：将单张 RGB 图像的三维场景重建形式化为 **Image-to-Code (Image2Code)** 问题：$P = \pi(I; \mathcal{E})$，$S = \operatorname{Exec}_{\mathcal{E}}(P)$，其中 $\mathcal{E}$ 为 Blender 5.0.1 + Blender-MCP 环境，智能体通过多轮「编辑代码 → 执行 → 渲染并检视结果 → 修订程序」产生轨迹 $\tau = \{(P_t, S_t, o_t)\}_{t=1}^T$，最终交付 `scene.blend`、`scene.glb` 与 `final.png`。
- 论文由四部分组成：
  1. **LEGO-Anything**：通用编码智能体在 Blender 中进行单图场景程序构建的框架。
  2. **LEGO-Bench**（§4, Appendix D–E）：基于 LychSim 与 Fab 商业合规资产构建的仿真真值基准。包含 208 张 RGB 输入图，源自 8 类环境、17 个主题下的 104 个逻辑场景，使用 443 个经规范化注册的资产（经 15 轮人工审核确认 368 个静态资产、75 个可动关节资产，含 327 条关节需求）。嵌套难度分层 $\mathcal{O}_E \subset \mathcal{O}_M \subset \mathcal{O}_H$，另含 10 个 NYC 航拍（Bird's-Eye）压力测试样本。
  3. **LEGO-Plugin**（§5, Appendix G）：针对轨迹退化诊断设计的免训练运行时插件，通过 MCP 工具、workflow skills 与 runtime hooks 接入原生 Harness，不改变规划器主体且不接触私有真值。
  4. **LEGO-World**（§6, Appendix H）：在不做任务微调的前提下，对 GPT-6-astra 重建冻结的场景执行确定性投影/深度渲染读出，评测 COCO val2017 检测、LVIS v1 val 实例分割与 ETH3D 相对深度。

### 2. 核心指标定义与校验细节（§4.2, Appendix E）
- **Validity ($V_i$)**：交付物须包含可重新加载且带至少一个网格和激活相机的 `scene.blend`、非空格式合法的 `scene.glb` 以及可解码非退化的 `final.png`，且主评测流程未发生未解报错；任一失败则 $V_i = R_i = A_i = 0$。
- **Reconstruction ($R_i$, F@5%)**：不做任何全局或逐物体配准/缩放，直接将提交场景与私有深度反投影点云在参考相机坐标系下对比。按私有实例掩码 $M_o$ 在图像平面分配归属（不依赖物体名称匹配），采用深度比例容差 $\tau(g) = 0.05\,z(g)$ 计算逐实例 F1 再宏平均（全局上限采样 100,000 点；另有 F@2%、F@10% 辅助诊断）。
- **Appearance ($A_i$)**：评测器保留提交场景的相机、几何、材质与灯光但固定渲染引擎与分辨率重新渲染 $\hat{I}_i$（不采用代理商提交的 `final.png`），统计 8-bit sRGB 下三通道最大误差 $\le 30$ 的像素占比。
- **Overall ($S$)**：$S = \frac{1}{N} \sum_{i=1}^N \frac{1}{2}(R_i + A_i) \cdot V_i$，失败任务保留在分母中记 0 分。人工盲评一致性校验（6 名标注员、240 个样本 / 960 个局部区域）显示标注员兼容一致率 86.6%，人工偏好与指标兼容性达 83.7%（$\varepsilon = 0.05$）。

### 3. 主实验、诊断与消融关键数据（§4.3, §5, §6, Appendix B, F, H）
- **LEGO-Bench 主表（Table 3，3 次运行均值±标准差）**：
  - `GPT-6-astra + Codex`：Indoor V=100.0 / R=52.4 / A=54.4 / S=53.4%；Outdoor V=98.0 / R=34.0 / A=45.5 / S=39.6%。
  - `GPT-6-sol + Codex`：Indoor S=32.3%；Outdoor S=24.2%。
  - `GPT-6-luna + Codex`：Indoor S=23.2%；Outdoor S=17.3%。
  - `GPT-5.6-sol / terra / luna + Codex`：Indoor S 分别为 14.8% / 15.4% / 14.1%；Outdoor S 分别为 15.3% / 11.5% / 12.1%。
  - 专用基线：`Gen3DSR` Indoor R=65.4% 但 A=0.0%（S=30.1%），Outdoor R=35.0%、A=0.0%（S=14.0%）；`VIGA`（纯程序模式）Indoor S=15.6%，Outdoor S=12.6%；`3D-RE-GEN` Indoor S=6.3%；`SceneGen` Indoor S=3.4%、Outdoor S=0.5%；`REST3D` Indoor S=2.8%；`SceneConductor` Indoor S=0.3%。
- **复杂度与推理预算（Table 4, Figure 4）**：所有 GPT 模型跨 Easy/Medium/Hard 平均 Validity 维持 99.5%，但综合分从 24.6% 降至 21.3% 再至 20.5%（Outdoor R 从 18.4% 降至 12.7%）。在 42 个 Office 子集上把推理强度从 Low 提到 XHigh，GPT-6-astra 从 32.3% 升至 61.8%，GPT-6-sol 从 21.3% 升至 39.7%，GPT-6-luna 从 14.4% 升至 21.2%，GPT-5.6 无稳定提升。
- **失败诊断（§5.1, Table 8）**：
  1. 初始化滞后：`GPT-6-astra` 约用 10% 预算得到首个可评测场景，`GPT-5.6-sol` 需约 20%。
  2. 迭代回退：`GPT-5.6-sol` 有 29.6% 的编辑降低分数，终态比历史最佳中间状态低 3.2 分。
  3. 自评不可靠：6×6 构建者-裁判矩阵（360 对检查点、2160 次主判定）中，自评在 Reconstruction 方向一致率仅 45.8%（跨模型 45.4%，低于随机 50%），Appearance 为 62.2%（跨模型 63.9%）。
- **LEGO-Plugin 增益（§5.3, Figure 8, Office 42 案例）**：`GPT-5.6-luna` +62.7%、`GPT-5.6-terra` +55.8%、`GPT-5.6-sol` +55.3%、`GPT-6-luna` +27.5%、`GPT-6-sol` +12.1%、`GPT-6-astra` +2.1%。三模块：Enhanced Initialization（VGGT 重力对齐 Z-up 曼哈顿框架 + 房间/相机协同初始化）、Grounded Refinement（SAM 3 区域掩码 + Depth Anything V2 相对深度显式残差校验，候选最多 2 轮修正）、Version Control（事务快照 + 保护实体白名单 + 最多 8 次事务 / 2 次回滚上限）。
- **LEGO-World 下游读出（§6, Table 5, Table 9）**：100 张图上（等置信度 AP 协议；94/100 有效检测/分割、99/100 有效深度），COCO BBox AP 30.14（对比 DINO 59.88）、LVIS Mask AP 14.75（对比 SAM 3 53.96）、ETH3D 相对深度 AbsRel 0.1554（对比 Depth Anything 3 0.0783）。
- **关节先导评估（Appendix B.4）**：Office 单案例试评在完备规范关节清单（14 可动 + 12 静态对照）下 $S_{\mathrm{Art}}^{\mathrm{pilot}} = 0$，表明当前单图场景重建尚不能可靠恢复可动关节结构。

### 4. 开源与运行基础设施核查（Appendix I, 项目页）
- 实验基于 Harbor Framework 扩展包 `lego_bench_harbor`（Docker/Compose + Blender 5.0.1 + Xvfb + Blender-MCP），模型通过 `amazon-bedrock` 提供，亦在基础设施层适配了 Codex 与 Hermes Agent（`NousResearch/hermes-agent` `aa6f77596b`）。
- 官方项目页仅给出 `Code · Coming soon` 与 `Early Access` 链接，未提供公开源码仓库地址，故无独立 `sources/repos/paper-agentic-2609-36380.md` 源码分析归档。
