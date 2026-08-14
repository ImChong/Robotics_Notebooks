# rift_wam_arxiv_2608_11521

> 来源归档（ingest）

- **标题：** Keep the Future, Drop the Rollout: Rift for World Action Models
- **短名：** Rift / RIFT
- **类型：** paper
- **来源：** arXiv abs / PDF
- **原始链接：**
  - <https://arxiv.org/abs/2608.11521>
  - <https://arxiv.org/pdf/2608.11521>
- **作者：** Chushan Zhang, Jinguang Tong, Xuesong Li, Yikai Wang, Hongdong Li
- **机构：** 澳大利亚国立大学（ANU；与同作者 EvoScene-VLA arXiv:2605.21862 一致；本预印本正文未单列单位行）
- **版本：** arXiv:2608.11521（2026-08）
- **入库日期：** 2026-08-14
- **一句话说明：** 用闭环干预证明 WAM 动作专家读的是 **位置绑定的未来 K/V**，不是去噪轨迹；**Rift** 用 anticipation token **一次** 写出完整未来 cache，部署不再滚视频。LIBERO **98.8% / 247.9 ms**（约 1.1× current-only）。**确认未开源**。

## 核心摘录

### 1) 问题
- Joint / IDM 类 WAM 在测试时 **迭代视频 rollout**，延迟是 current-only 的 **3.3×–9.6×**。
- Fast-WAM / PFD 等把未来读 **拿掉或蒸馏掉**，延迟下来但相对显式读未来仍有缺口。
- 既有对比同时改「有没有未来表征」和「怎么构造它」，分不开 **消费** 与 **生产**。

### 2) 干预协议（Finding）
- 对象：四套 WAM（Fast-WAM-Joint / IDM、Cosmos-2、LingBot-VA）在 **全部 40 个 LIBERO 任务**、每干预 **2000** 成对 trial。
- 视频 token 不能 attend 动作 → 未来 cache **与动作无关**，可 record/replay。
- **Mask 未来读：** EE-ADE **18.7 cm**，成功率 98.4%→**9.7%**。
- **空间打乱 / 时间对调：** ADE 相近（14.3 / 15.6 cm）但成功率差很大（65.2% / 0.7%）→ 读的是 **指定位置上的内容**。
- **Final-clean K/V 重放**（Joint / Cosmos-2）：ADE **1.9 / 1.7 cm**，成功率 **97.9 / 98.2%** ≈ Original。IDM / LingBot-VA 本来就读一次干净 cache，该干预结构上 N/A。
- 结论：消费侧 **一份固定干净 cache 就够**；生产侧仍靠迭代 rollout——这才是 Rift 要换的。

### 3) Rift 方法
1. 保留 Fast-WAM-Joint 的未来读接口与 Wan2.2-5B 骨干。
2. 把未来 token 换成可学习 **anticipation tokens** \(E\in\mathbb{R}^{m\times d}\)，继承时空下标；满对齐 \(m=n(T_{\mathrm{lat}}-1)\)（LIBERO **m=196**，RoboTwin **m=240**）。
3. 一次 VideoStack prefill：\([f_0(o);E]\) → 每层未来位置 \((K_E,V_E)\)；动作流全程复用同一 cache。部署 **无视频扩散、无 VAE 解码**。
4. 训练两路前向：原生视频 flow \(\mathcal{L}_{\mathrm{vid}}\)；部署对齐的 \(\mathcal{L}_{\mathrm{act}}\) + 条件 FM \(\mathcal{L}_{\mathrm{FM}}\) + stopgrad L2 probe。后期对第一帧加噪只监督辅助头，**动作损失只用干净行**。
5. 可选 shadow：probe 与 FM 样本的归一化分歧作 CUSUM 预警（失败平均提前 **210** 步）；**不进** 策略延迟数字。

### 4) 实验（论文报告摘要）

| 方法 | 未来读 | Rollout | LIBERO SR | 延迟 (A800, ms/chunk) |
|------|--------|---------|-----------|------------------------|
| Fast-WAM | × | × | 96.8±0.27 | 235.7（1.0×） |
| PFD | × | × | 97.3±0.12 | 257.0 |
| Fast-WAM-Joint | ✓ | ✓ | 98.4±0.26 | 780.2（3.3×） |
| Fast-WAM-IDM | ✓ | ✓ | 98.6±0.34 | 1081.2 |
| LingBot-VA | ✓ | ✓ | 98.5±0.08 | 2270.3（9.6×） |
| **Rift** | ✓ | × | **98.8±0.17** | **247.9（1.1×）** |

- **LIBERO-Plus**（10,030 变体、不训练）：Rift **81.1%**，相对 Fast-WAM-IDM **+9.7 pt**。
- **RoboTwin 2.0** Clean/Rand：**92.9 / 92.6**（评测集最高）；PFD 92.5/92.1；LingBot-VA 92.4/91.4。
- **消融：** Rift-L2 98.37%；条件 FM 98.8%；\(m\) 从 2 增到满对齐都高于 Fast-WAM 的 96.75%。

### 5) 开源核查（步骤 2.5）
- **无独立项目页**（用户未给；arXiv HTML 亦无 Code 链）。
- **GitHub 检索**（2026-08-14）：无官方 `Rift` / `smpc2rl` 类匹配仓。
- **结论：** **确认未开源**。wiki 时序图标不适用。复现坐标仍是 [FastWAM](https://github.com/yuantianyuan01/FastWAM) 骨干描述，不是可跑 Rift 实现。

## 对 wiki 的映射

- 升格 [Rift 论文实体](../../wiki/entities/paper-rift-wam.md)
- 更新 [World Action Models](../../wiki/concepts/world-action-models.md)、[WAM 动作后果 01](../../wiki/overview/wm-action-consequence-category-01-wam-action-prediction.md)、[DreamWAM](../../wiki/entities/paper-dreamwam.md)、[VLA 部署指南](../../wiki/queries/vla-deployment-guide.md)

## 当前提炼状态

- [x] 干预发现 + Rift 接口 + 三张主表 + 开源边界
- [x] wiki 实体页与交叉引用
