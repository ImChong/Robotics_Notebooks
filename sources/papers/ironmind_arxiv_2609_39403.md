# IronMind：基于相机坐标系第一人称预训练的人形机器人灵巧操作

> 来源归档（paper · arXiv:2609.39403v1 全文核对）

- **标题：** IronMind: Scaling Humanoid Dexterous Manipulation via Camera-Space Ego-Centric Pretraining
- **作者：** Huimin Pan\*, Yufan Ren\*†, Kunpeng Song\*, Siyang Wang\*, Xiwen Zhang\* 等 19 位作者（World Model Team, XPENG Robotics；† Project lead；‡ Chenyi Chen Supervision）
- **机构：** 小鹏机器人（XPENG Robotics）
- **年份：** 2026（arXiv:2609.39403v1，2026-09-30）
- **arXiv：** <https://arxiv.org/abs/2609.39403>
- **HTML 全文：** <https://arxiv.org/html/2609.39403>
- **项目页：** <https://xpeng-robotics.github.io/ironmind/>（[`sources/sites/xpeng-robotics-ironmind.md`](../sites/xpeng-robotics-ironmind.md)）
- **入库日期：** 2026-10-02
- **核对范围：** 通读 `arXiv:2609.39403v1` HTML 全文（§1–6、Table 1–5、Fig. 1–10）及项目页 <https://xpeng-robotics.github.io/ironmind/>，核查开源状态（未公开代码、权重与数据集）。未运行模型或真机测试。
- **通读：** 已完成
- **一句话说明：** 把人类第一人称视频和异构机器人动作统一到头部相机坐标系，经四阶段数据清洗与 10,000+ 小时预训练构建双专家 MoT VLA（可选多模态世界先验训练），在 IRON-R01 真机六项 OOD 任务上达到 55.0% 成功率。

## 核心论文摘录（arXiv:2609.39403v1）

1. **相机坐标系动作表示（§3.1.1）：** 低成本头戴第一人称视频缺乏躯干/脊柱运动捕捉，强制重定向到机器人躯干系会引入刚体脊柱假设与挂载外参错配误差。IronMind 保留人类 3D 手部与相机 SLAM 轨迹在相机坐标系的原生表示，利用外参标定将异构机器人轨迹（预训练不含 IRON-R01）变换至头部相机系，并按语义对齐腕部与手指屈曲维度（无需显式全身/手部运动学重定向或隐式动作瓶颈）。
2. **10,000+ 小时数据策展管线（§3.1.2）：**
   - 语料覆盖家庭（最大占比）、餐饮、实验室/车间、酒店、零售、教育、办公等场景，总计 >10,000 小时第一人称人类视频与非目标本体异构机器人数据。
   - **四步清洗：** (a) 规则过滤缺失/异常相机内参与手姿追踪；(b) 切分单意图片段并用 `Qwen3.6-35B-A3B` 重标原子指令（段边界掩码动作 loss）；(c) 速度分布对齐（时间重缩放使敏捷人手轨迹接近遥操速度）；(d) 逐帧质量加权 $p(f)=e(f)g(f)\cdot s_{\mathrm{cam}}(f)s_{\mathrm{move}}(f)b_{\mathrm{vel}}(f)$ 并做逆变换采样。
3. **双专家 MoT 架构与可选世界先验（§3.2–3.3）：**
   - **Understanding Expert**（预训练 VLM 骨干处理头摄 RGB 与指令）+ **Action Expert**（flow-matching DiT 预测连续相机系动作块），在指定层通过流特定投影的共享注意力融合。
   - **可选多模态世界先验（仅训练期）：** 理解专家前缀加入可学习 query token，分别接冻结 `Depth-Anything-3-Base`（几何深度）、`DINOv3`（外观不变语义）与冻结 `Wan2.2` 视频生成器（0.4 s 未来 VAE latent 动力学，亦对比 3D motion flow）。**推理部署时移除全部先验头与教师模型**，仅保留 query token。
   - **两阶段训练：** 预训练（AdamW, global batch 1024）→ 后训练（约 **40,000** 条过滤 IRON-R01 遥操轨迹，外参转换至相机系，双专家联合微调，**10 Hz** 真机闭环）。
4. **缩放规律与多层评测（§4, Table 1–5）：**
   - **验证损失缩放：** 从 250 h 扩到 10,000 h，验证 loss 呈对数线性下降，斜率 **−0.058 / decade**，至 10,000 h 仍未饱和。
   - **开环（970 approaching + 120 grasping，Table 1）与策略闭环（3 chunks，Table 2）：** 250 h → 10,000 h，开环 $d_{p10}$ 从 37.8→**25.8 cm**、Progress 14.7%→**44.9%**、$\theta_{\mathrm{align}}$ 47.8°→**26.0°**、$e_{\mathrm{FA}}$ 19.97→**15.66 mm**；3-chunk 闭环 $d_{p10}$ 32.0→**14.6 cm**、Progress 21.4%→**49.3%**；盲测人工胜率 10,000 h 对 250 h 为 **67.1%**（47/70 有效票，30 平局）。
   - **世界先验消融（Table 4，开环）：** 三先验（0.4 s）相对无先验 $d_{p10}$ 27.6→**22.6 cm**、Progress 40.9%→**53.4%**；但**真机闭环（Table 5）反而是无先验 55.0%（progress 71.3%）高于带世界先验 31.7%（53.3%）**，作者推测辅助预测目标与接触丰富闭环控制存在对齐失配。
   - **IRON-R01 真机六项 OOD（Semantic / Affordance / Reasoning，各 10 trials，Fig. 10 & Table 5）：** 0h 无预训练 **5.0%**，≤5,000h 最高 **11.7%**，10,000h 跃升至 **55.0%**（progress 71.3%）；同预算躯干坐标系基线仅 **26.7%**（46.2%），InternVLA-A1.5 微调为 **30.0%**（47.8%）。
5. **局限（§5）：** 语料集中在家庭/厨房/车间；仍依赖相机外参标定与 IRON-R01 真机后训练；真机评测仅限单本体、6 项 OOD、每任务 10 次；世界先验开环增益未能稳定转化为闭环增益。

## 开源状态

截至 2026-11-22，官方项目页与 `xpeng-robotics` 组织仅提供论文、图表与真机演示视频，**未公开代码、模型权重或数据集下载链接**。

## 对 wiki 的映射

- 实体页：[`wiki/entities/paper-ironmind.md`](../../wiki/entities/paper-ironmind.md)
- 项目页归档：[`sources/sites/xpeng-robotics-ironmind.md`](../sites/xpeng-robotics-ironmind.md)
- 与 [EgoScale](../../wiki/methods/egoscale.md)、[VLA](../../wiki/methods/vla.md) 对照
