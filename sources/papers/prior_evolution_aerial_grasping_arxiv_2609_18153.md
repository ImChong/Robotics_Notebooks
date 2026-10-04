# Prior Evolution and Task Alignment for Aerial Grasping（arXiv:2609.18153）

> 来源归档（ingest）

- **标题：** Prior Evolution and Task Alignment for Aerial Grasping
- **类型：** paper / aerial grasping / aerial manipulation / trajectory optimization / cross-entropy method / execution-aware learning
- **arXiv abs：** <https://arxiv.org/abs/2609.18153>
- **PDF：** <https://arxiv.org/pdf/2609.18153v1>
- **版本：** v1，2026-09-16 提交；20 页、14 幅图（arXiv 元数据）
- **作者：** Weiliang Deng、Zhengyang Dang、Yao Mu、Ximin Lyu
- **代码 / 数据：** 截至 2026-10-04，在 arXiv 摘要、作者/研究页面及公开检索中未发现独立项目页、官方代码仓或数据集链接；这表示“未发现公开入口”，不等同于作者明确声明永不开源。
- **入库日期：** 2026-10-04
- **一句话说明：** 针对空中抓取轨迹优化的非凸与初始化敏感问题，学习并通过 CEM 进化轨迹先验；再让 Execution-Aware Critic 从接触、抬升、任务完成等执行结果中学习成功概率，并把冻结 critic 的能量作为可微抓取代价反向塑造轨迹。

## 摘要级要点

- **问题一：优化脆弱。** 空中抓取的轨迹优化具有非凸性，有限计算预算下，初始轨迹会显著影响能否找到高质量解。
- **问题二：代理目标与任务成功不完全一致。** 手工设计的数值目标只表达部分可计算条件，未必覆盖真实物理执行中的接触、稳定抓取与完成。
- **先验进化：** 先从已优化轨迹学习 trajectory prior；CEM 从先验采样初始轨迹，经部署时使用的优化器求解，再保留表现较好的样本作为新监督，迭代改进先验。
- **执行对齐：** Execution-Aware Critic 以接触、抬升、完成等结果学习轨迹在物理执行中成功的可能性。
- **反哺优化：** 将训练完成并冻结的 critic 能量作为可微抓取代价，令执行数据参与轨迹生成，而非仅作为事后评估。
- **证据范围：** 摘要报告仿真与真实世界实验改善优化可靠性、轨迹一致性和抓取表现；本归档不补写摘要未提供的指标或硬件细节。

## 方法摘录（摘要级）

### 1. Prior Evolution：围绕部署优化器选初值

这里的 CEM（Cross-Entropy Method）并不是直接取代轨迹优化器，而是作用于其初始化分布：

1. 轨迹先验给出一批候选初始轨迹。
2. 每个候选都通过实际部署所用的轨迹优化器求解。
3. 依据候选优化结果筛出较优轨迹。
4. 把筛选结果纳入新监督，更新轨迹先验后再采样。

重要的对齐点是“先验质量”按部署优化器优化之后的结果衡量，而非仅按初始样本的表面质量衡量。

### 2. Execution-Aware Critic：把结果监督带入代价

Critic 学习与物理执行结果相关的评价信号，摘要明确点出接触、抬升和任务完成。其冻结后的能量被用作可微抓取 cost，因此它不只是排序器：梯度可用于引导轨迹生成，使学习到的任务成功线索回流至解析优化过程。

## 评测摘要与解读

- **实验范围：** 论文摘要称包含仿真和真实世界实验。
- **报告方向：** 优化可靠性、轨迹一致性、抓取表现均有改善。
- **不要过度解读：** 摘要没有给出可核实的绝对成功率、基线名称、试验次数、机器人型号或计算开销；定量结论需回看 PDF 正文与补充材料。
- **复现状态：** 未检索到官方代码或数据入口，无法据此复建训练、CEM 采样或真机评测管线。

## 对 Wiki 的映射

- [论文实体页](../../wiki/entities/paper-prior-evolution-aerial-grasping.md)
- [Manipulation 任务页](../../wiki/tasks/manipulation.md)
- [Trajectory Optimization 方法页](../../wiki/methods/trajectory-optimization.md)

## BibTeX

```bibtex
@article{deng2026priorevolution,
  title={Prior Evolution and Task Alignment for Aerial Grasping},
  author={Deng, Weiliang and Dang, Zhengyang and Mu, Yao and Lyu, Ximin},
  journal={arXiv preprint arXiv:2609.18153},
  year={2026},
  doi={10.48550/arXiv.2609.18153}
}
```

## 当前提炼状态

- [x] 核对 arXiv 标题、作者、版本与摘要
- [x] 检索既有知识库 arXiv 节点，未发现重复项
- [x] 检查公开项目/代码入口，未发现官方仓库链接
- [ ] 定量表格与硬件细节待逐项核对论文 PDF 正文
