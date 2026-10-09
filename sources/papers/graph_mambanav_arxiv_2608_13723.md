# Graph-MambaNav: Spatial-Temporal Graph Mamba Leveraging Object-Relation Knowledge for Object-Goal Navigation

> 来源归档（paper / arXiv v1）

- **标题：** Graph-MambaNav: Spatial-Temporal Graph Mamba Leveraging Object-Relation Knowledge for Object-Goal Navigation
- **类型：** paper（arXiv:2608.13723v1）
- **作者：** Leyuan Sun, Genxin Chen, Linwei Ye, Yan Zhang, Xi Kan, Yanfei Sun
- **机构：** 无锡学院物联网工程学院；南京邮电大学通信与信息工程学院（Yan Zhang）；无锡市人工智能与安全重点实验室（Yanfei Sun）
- **提交日期：** 2026-08-13
- **发表状态：** arXiv 页面注明已接收 IEEE Robotics and Automation Letters（RA-L），并将转至 ICRA 2027 展示
- **论文：** <https://arxiv.org/abs/2608.13723v1>
- **HTML 正文：** <https://arxiv.org/html/2608.13723v1>
- **作者主页：** <https://leyuan-sun.github.io/>
- **IEEE 条目 / Demo：** <https://ieeexplore.ieee.org/document/11676043>
- **入库日期：** 2026-10-09
- **一句话说明：** 针对物体目标导航，把目标相关性转成图节点扫描顺序，并以局部关系传播、空间 Graph-Mamba 与逐物体时间 Mamba 共同驱动策略。
- **源码开放核查：** 截至 2026-10-09，arXiv 论文页和作者主页没有列出该论文的官方 GitHub 或可运行代码仓库；暂不标记为已开源。作者主页提供 IEEE 条目与演示入口。

## 核心摘录（面向 wiki 编译）

### 摘录 1：问题与设计判断

- ObjectNav 智能体只接收第一人称 RGB 图像，在陌生室内寻找指定物体；目标物体常较小或暂时不可见，需要利用「遥控器常在电视附近」一类关系线索。
- 既有物体图方法多在特征融合或注意力权重里编码目标相关性，但图计算顺序本身通常不受目标控制。Graph-MambaNav 的关键主张是：在因果 Mamba 扫描中，让越相关的对象越晚处理、目标节点最后处理，从而能汇聚更多前序上下文。
- 这不是简单把 GNN 替换成 Mamba：关系先验同时初始化局部边属性与全局节点顺序。

**对 wiki 的映射：** object-goal-navigation、object-relations、graph-mamba

### 摘录 2：模型结构与数据流

- 输入为每步单目 egocentric RGB 与目标类别。冻结的 COCO 预训练 DETR 取每类最高置信度检测；共 22 个物体类别。ResNet-18 提取全局视觉特征，GloVe 编码类别文本，DETR 提供外观特征、框和置信度。
- 作者用 ChatGPT-5 按目标条件化生成 22×22 物体关系亲和矩阵；这是 commonsense 先验的来源。论文描述将它用于初始化图边属性与节点排序，没有说每个导航时刻都在线调用 LLM。
- 空间编码并行执行两条路径：GINE 做带边属性的局部邻居消息传递；全局路径依亲和度从低到高排序，目标节点置于末尾后执行 Graph-Mamba selective scan；两路相加后经 FFN 融合。
- 时间模块维护最多 35 帧、按类别对齐的 FIFO 记忆，并对每个物体的历史序列运行 Mamba。目标文本再作为 query 对该时空记忆做 cross-attention，最后与当前 ResNet 视觉特征和上一动作融合，送入 LSTM-A3C 策略。

**对 wiki 的映射：** graph-mamba、mamba、a3c、objectnav

### 摘录 3：仿真实验与主要指标

- **AI2-THOR：** 120 个室内场景，厨房/客厅/卧室/浴室各 20 训练、5 验证、5 测试；22 类目标。全轨迹 SR/SPL 为 **83.22% / 46.52%**；最短路径长度 $L\geq5$ 子集 SR/SPL 为 **76.09% / 46.20%**。
- **RoboTHOR：** 89 套公寓；文中列出 60 训练、5 验证、10 测试环境（合计 75，未解释其余 14 套），12 类目标。全轨迹 SR/SPL 为 **49.82% / 28.67%**；$L\geq5$ 子集为 **37.38% / 22.49%**。
- 论文称每组指标取 3 次测试运行均值与标准差；实现使用 A3C 训练数百万 episode、3 张 RTX 3080 Ti，策略为两层 512 hidden 的 LSTM。基线取自原论文中相同或可比协议；Memory-MambaNav 则由作者重实现且去掉其额外 reward 设计，因此跨方法比较仍需留意协议差异。
- 在 AI2-THOR 长路径上，论文报告 Graph-MambaNav 的优势更明显；去掉时间扫描后，$L\geq10/15/20$ 的 SR 为 **61.23/41.02/27.32%**，完整方法为 **65.92/42.63/33.40%**。
- 模型效率对比使用 T=35：论文报告 Graph-MambaNav **12.58M 参数、23.47 GFLOPs、2.53 GB 显存、22.12 FPS**；对应 Transformer 变体为 **62.13M、82.39 GFLOPs、6.72 GB、11.32 FPS**。这些数字属于论文的特定实现与评测环境，不可直接当作任意硬件上的端到端导航频率。

**对 wiki 的映射：** objectnav、ai2-thor、robothor、sr、spl

### 摘录 4：组件消融、真机与局限

- AI2-THOR 消融中，单加局部扫描 SR 从 63.83% 升至 66.15%；再加目标感知全局扫描升至 78.34%；再加时间扫描达到 83.22%。这支持了“计算顺序”和“逐物体历史”都产生作用，其中全局有序扫描贡献最大。
- 关系先验不具普遍可靠性：作者指出 light switch 等关系区分度低时会引导偏置探索；RoboTHOR 的 pot 示例中，实际布局也可能与常识亲和矩阵不符。给矩阵加入 30% 随机噪声后，AI2-THOR SR 从 83.22% 降至 77.92%。
- 真机验证使用带 0.65 m 高度单目相机的轮式机器人，底盘与相机俯仰舵机负责运动；Raspberry Pi 5 运行 ROS 2 低层控制，导航模型在外部主机上运行并通过局域网 ROS 2 topic 通信。论文展示在客厅寻找一本书的单次路线案例，没有给出多场景成功率或真机量化基准。

**对 wiki 的映射：** 工程验证、先验鲁棒性、sim2real

## 对 wiki 的映射

- 升格实体页：[wiki/entities/paper-sa-2608-13723-graph-mambanav.md](../../wiki/entities/paper-sa-2608-13723-graph-mambanav.md)
- 任务页回链：[wiki/tasks/zero-shot-object-navigation.md](../../wiki/tasks/zero-shot-object-navigation.md)

## 当前提炼状态

- [x] 核对 arXiv v1 HTML 正文、作者主页、发表状态与项目/代码入口
- [x] 提炼方法数据流、基准、指标、消融、真机设置与局限
- [x] 新建唯一论文实体，并更新 ObjectNav 任务页交叉引用
