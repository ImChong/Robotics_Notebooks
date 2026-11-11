# LawKAN: Group-Law Layers for Composition — When Imposing a Known Law Pays Off（arXiv:2611.12567）

> 来源归档（paper）

- **标题：** LawKAN: Group-Law Layers for Composition — When Imposing a Known Law Pays Off
- **类型：** paper
- **作者：** Anna Mishchenko, Nikolai Gavrilin（Independent researchers）
- **arXiv：** <https://arxiv.org/abs/2611.12567>
- **PDF：** <https://arxiv.org/pdf/2611.12567v1>（v1，2026-11-09，cs.LG）
- **代码：** https://github.com/occamolab/lawkan （AGPL-3.0-or-later；论文 Data and code availability 与附录 A.1）
- **入库日期：** 2026-11-11
- **一句话说明：** 把已知群运算（如 SU(2) 的 Hamilton 积）精确放进 KAN 式层的节点，边只学局部增量系数，另配零初始化修正 ψ 学习「法则本身的偏差」；在合成组合任务与真实机械臂标定 / IMU 上给出「何时赢、平、输」的实测地图。
- **核对范围：** 读完 PDF 正文第 1–8 节（含全部表 1–6、图 1–4 的图注与数据）；附录读了 A（复现与优化协议）、C（精确可表示性证明、解析支撑、截断 BCH）、E.3（边界压力测试）、F.1–F.2（IMU 与陀螺标定）、G.4（几乎成立的律、识别性与隔离训练）、K（预注册预测裁决）、L–M；附录 B、D（逐 seed 表与跨群证明）、E 其余部分、F.3、G.1–G.3、H、I、J 仅按标题与图表浏览，未逐段读，其中的逐 seed 数字与推导未独立核对。未运行代码，未下载仓库代码，只静态读了仓库 README。
- **通读：** 已完成

## 开源状态

- **已开源**：仓库 <https://github.com/occamolab/lawkan>（论文称含 canonical.json 清单、复现脚本与 `tools/span_check.py`）；README 可访问，内容与论文摘要一致。代码实际能否复现未运行验证。
- 数据集多为公开数据（BROAD、MUKCa、FR3/PhilNad、IPA Series A 等，许可见论文末尾）；ABB RobotCali 无明确许可，作者称经授权使用、不再分发。

## 核心摘录

1. **方法（第 1、3.2 节）：** 层 = 固定特征 + 可学习边系数 + 无参数精确节点。SU(2) 情形：特征 {1,cos α,sin α}⊗{1,cos β,sin β}（每因子 9 个标量），边 W_i∈R^{3×9}（n=2 时共 54 参数）输出 so(3) 旋转向量，节点 exp 后做 Hamilton 积，读出为旋转探针向量的方向。Lemma 1：固定旋转量时存在权重使网络对所有输入精确复现组合（构造性证明，附录 C）。
2. **两个判据（第 3.3 节）：** Criterion 1（span）线性边 + 精确节点可精确表示组合，当且仅当特征行空间包含 log χ 的某一支（至双覆盖对径对称），否则存在与训练无关的误差下限（证明见附录 C.1）；Criterion 2（law-match）若真实律为 A⋆B=(AB)exp(η(A,B))，仅用边时下限一阶为 ‖η‖，ψ 在 η 落入其特征张成时可消除。
3. **修正 ψ（第 3.4 节）：** R(A,B)=(AB)exp ψ(f(A),f(B))，末层零初始化；右乘在物理复合（一个关节 / 一步）上，不在长链每个中间积上做；默认「全局线性 + 局部 RBF 核」两槽头；单独冻结精确边训练（隔离）以避免与边互相补偿。
4. **合成基准（第 4.2.1 节，表 2）：** 固定角 SU(2) 复合，噪声 σ=0.03，LawKAN-SU(2) 54 参数 2.78±0.02°（5 seed），MLP+四元数头 h=32（1476 参数）3.36±0.06°；最强调参基线 MLP-Quat h=64 为 3.25±0.02°（4996 参数）。摘要称比最强固定矩阵基线少 27× 参数、距 Monte-Carlo Bayes 下限 +0.05°。
5. **真实 FR3 机械臂标定（第 5.1 节，表 3）：** 114 个真实位姿（Nadeau, 2024 数据集，编码器 + 相机恢复的法兰位姿），5 个随机划分 seed，末端位置误差。两槽 ψ（306 参数，隔离训练）N=20/40/60/80 为 1.51/1.24/1.14/1.12 mm；Gauss–Newton 强参考（54 参数）1.56/1.39/1.29/1.33 mm；精确节点 1.53/1.49/1.37/1.40 mm；残差 MLP（验证选择，18k–69k 参数）2.39/2.03/1.35/1.33 mm。vs 精确节点配对 p=0.84/0.028/0.007/0.003（N=20/40/60/80）。
6. **跨厂商（第 5.3 节，表 4）：** MUKCa 插座一致性协议，KUKA iiwa14 与 Panda 上 ψ 胜过缩减边（p<0.001）并优于稠密螺旋参考；FR3 unit 上稠密螺旋已张成失配，ψ 不能超过（0.29 mm）；Kinova 上全局槽 2/10 seed 崩塌，局部核头避免。
7. **范围与失败（第 4.2.3、6 节）：** 目标无群律时 Hamilton 节点 23.87±0.41° 对 MLP h=4 的 12.64±3.00°；剪枝边仅适用固定已知角；链长可扩展到 n=4；n=6 起 seed 双峰、n=8 全部接近随机。

## 对 wiki 的映射

- [paper-lawkan](../../wiki/entities/paper-lawkan.md)
