# RealtimeWAM 论文来源归档

## 元数据

- **题目：** RealtimeWAM: How Fast Can I Run My World Action Model?
- **作者：** Huanan Liu、Ye Li、Kangye Ji、Xiaoyu Chen、Hanyun Cui、Yutian Shen、Yuan Meng、Chenglei Wu、Jingyan Jiang、Bo Li、Zhi Wang
- **单位：** 清华大学；YuanxingGuangnian Robotics（作者栏原样，疑为「元行广念」类名称，未核实中文名）；南京大学。脚注：一位作者为清华实习期间完成，另有通讯作者
- **预印本：** https://arxiv.org/abs/2610.10079
- **HTML：** https://arxiv.org/html/2610.10079v1
- **版本：** v1，2026-10-07，cs.RO
- **项目页：** https://anonymous.4open.science/w/realtimewam/ （匿名页，归档见 [sources/sites/realtimewam.md](../sites/realtimewam.md)）
- **代码：** 截至 2026-10-12 未发布

## 论文要解决的问题

WAM 同时做未来世界预测与动作生成，视频生成使推理延迟高（实验中 FastWAM 约 214 ms、OpenWAM 约 673 ms），限制真机闭环响应。RealtimeWAM 是**免训练**框架：不改权重，通过并行执行与自适应计算降低延迟。

## 方法与训练信息

1. **被加速的两个骨干：** FastWAM（Yuan 等，arXiv:2603.16666，推理时省略未来视频生成，仅缓存动作 Transformer 残差）与 OpenWAM（Wang 等，arXiv:2609.07398，视频与动作联合预测，缓存视频与动作两个 Transformer 的残差）。后者是 NUS/清华等的 OpenWAM，**不是** Stanford 的 2610.07922。直接使用官方发布的微调检查点，不再训练。
2. **机制一：并行执行（层间依赖）：** 每层投影后立刻释放观测 K/V，使预测与 Prefill 余下层重叠；VAE 编码与条件准备并行；指令特征不变时复用；实现用融合算子、预分配缓冲与 CUDA Graph 重放。
3. **机制二：选择性复用。**
   - 稳定区域 token 复用（Prefill）：把 RGB 平均绝对变化超过 δ 的位置记为动态集 D_t，再按保留比例 α 补充；容量 r_t=|D_t|+⌈α(M−|D_t|)⌉，取不小于它的预剖析容量 b_t（附录剖析 {80,120}）；其余位置按首层特征相对变化排序；第 0 层全稠密，更高层仅对选中位置重算 FFN，其余用缓存，注意力与残差门始终全算，未来视频与动作 token 全算。主方法的 δ、α 数值正文未给出。
   - Transformer 残差缓存（运动自适应细化）：缓存块输入到输出的残差 C=B(h)−h，每个 Euler 步计算 Head(h+C, s)；10 个 Euler 更新中第 0、3 步刷新，之后解码粗动作块（H=32）；运动分数 m_t 为末端位置最大位移，预算 N_t=2+2·1[m_t≤τ]，τ=0.08 m（仅 XYZ）：运动小则 4 次完整前向，运动大则 2 次。
4. **运行环境：** BF16、batch 1、改装 48GiB RTX 4090、PyTorch 2.7.1 / CUDA 12.8、Core Ultra 9 285K；`reduce-overhead` 编译并验证图重放。准备阶段调度每请求节省 0.778 ms（VAE/条件重叠）与 0.274 ms（RGB 提前路由）。

## 论文报告的结果

- **主表（RTX 4090）：** FastWAM 原生 平均 SR 82.77%、214.33 ms → RealtimeWAM 82.75%、24.09 ms（8.90×）；OpenWAM 原生 87.94%、672.83 ms → 87.41%、63.09 ms（10.67×）。SR 平均为 RoboTwin Clean / Random、LIBERO、LIBERO-Plus 四项等权。与同样使用编译 + CUDA Graph 的原生相比，加速分别为 4.14× 与 7.15×。
- **与其他加速方法（OpenWAM 平均 SR / 延迟）：** VLA-Cache 86.10 / 400.26 ms；TeaCache 86.74 / 245.99；ToCa 87.30 / 440.42；BAC 87.77 / 305.08。FastWAM 上 VLA-Cache 77.20 / 115.55；TeaCache 79.63 / 49.28；ToCa 81.57 / 102.66；BAC 82.56 / 67.14。
- **累积执行优化（全请求，FastWAM / OpenWAM，ms）：** 原生 214.33 / 672.83；+CUDA Graph 104.09 / 524.69；+编译 99.68 / 450.90；+并行 33.52 / 223.27；+token 复用 29.47 / 217.24（10F 设置）。
- **组件消融（OpenWAM, RoboTwin）：** 完整 93.4% / 63.76 ms；去 token 复用 93.8% / 72.10 ms；随机 token 91.3%；去容量补充 92.1%；随机门控 91.1%；固定 2F 89.6% / 54.70 ms；固定 4F 93.7% / 95.26 ms。去掉 token 复用成功率略高（93.8 对 93.4），原文未作额外解释。
- **运动自适应细化（OpenWAM, 93 对观测）：** 固定 2F 平均 NFE 2.00、XYZ 偏差 7.65 mm；固定 4F 4.00、3.68 mm；自适应 2.65、6.48 mm。
- **真机（Franka Research 3 + RealSense D435i，20 Hz，每请求 32 动作块，每任务 50 次）：** 平均 SR FastWAM 35.6→52.8（延迟 210.71→23.72 ms）；OpenWAM 33.6→70.8（640.87→65.27 ms）。任务：三个羽毛球（24→40 / 4→68）、胡萝卜入盘（46→74 / 66→96）、挂杯（72→70 / 82→82）、网球入碗（26→34 / 16→76）、滚筒入碗（10→46 / 0→32）。传送带 1 m/min 的实验中，延迟超过约 90 ms 时成功率明显下降（经验阈值）。

## 复现与解释边界

- 论文无独立局限节；从内容看：组件消融仅在 OpenWAM + RoboTwin 上；细化分析仅 3 条轨迹 93 对观测；真机五个任务单一平台；90 ms 阈值为该任务的经验值，论文未给出单独的因果分析。
- δ、α 主方法数值正文未给出，仅凭页面无法复现稳定区域选择。
- 匿名项目页只有论文 PDF、示例视频与图，没有代码。

## Robotics_Notebooks 映射

- 详情节点：wiki/entities/paper-realtimewam-training-free-wam-acceleration.md
- 被加速对象：wiki/entities/paper-fast-wam.md、wiki/entities/paper-openwam.md

## 一手来源

- arXiv 摘要：<https://arxiv.org/abs/2610.10079>
- arXiv HTML：<https://arxiv.org/html/2610.10079v1>
- 匿名项目页：<https://anonymous.4open.science/w/realtimewam/>
