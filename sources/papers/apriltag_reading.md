# AprilTag 系列论文通读归档（2011 / 2016 / 2019）

> 来源归档（AprilTag 1 -> 2 -> 3 三篇原始论文通读记录）

- **标题：** AprilTag / AprilTag 2 / AprilTag 3（Flexible Layouts）学术论文合集
- **类型：** paper
- **机构：** APRIL Robotics Lab, University of Michigan（Edwin Olson, John Wang, Maximilian Krogius, Acshi Haggenmiller）
- **链接：**
  - AprilTag (ICRA 2011): <https://april.eecs.umich.edu/papers/details.php?name=olson2011tags>
  - AprilTag 2 (IROS 2016): <https://april.eecs.umich.edu/papers/details.php?name=wang2016apriltag>
  - AprilTag 3 (IROS 2019): <https://april.eecs.umich.edu/papers/details.php?name=krogius2019flexible>
- **核对范围：** 通读三篇原文全文（Olson ICRA 2011、Wang & Olson IROS 2016、Krogius et al. IROS 2019），重点核对编码族设计、检测器演进、位姿恢复、复杂度度量及 LabelMe / 实测定位评测数据。
- **通读：** 已完成
- **沉淀到 wiki：** [`wiki/entities/april-tag.md`](../../wiki/entities/april-tag.md)

## 三篇原文核心要点编译

### 1. AprilTag（Olson, ICRA 2011）

- **系统动机：** 相比面向高载荷存储的 QR 码（268+ 对准单元，识别距离受限），机器人基准标记仅需 4–12 bit 级唯一 ID 载荷（总单元约 49–100 格），借此换取更远检出距离、抗透视畸变与实时性。
- **检测与解码：** 基于图像梯度做 Felzenszwalb 风格并查集局部聚类拟合有向线段（暗左亮右），深度为 4 的 DFS 搜索闭合四边形（quad），经 DLT 求单应矩阵并结合焦距和 tag 物理尺寸分解旋转/平移（以正交极分解修正旋转矩阵）。解码采用边框已知黑白像素分别回归空间线性灰度模型 $I(x,y)=Ax+Bxy+Cy+D$，以其均值作自适应局部阈值。
- **编码设计（Lexicodes + 几何复杂度）：** 在经典字典码基础上增加四重旋转（0°/90°/180°/270°）最小 Hamming 距离约束，引入矩形覆盖复杂度阈值（如 `c >= 10`）与大素数步长遍历以提升比特熵；提出 `36h9`/`36h10`/`36h15` 等码族。
- **实验：** LabelMe 180,829 张自然图像测试验证复杂度启发式与 `36h10` 低于 ARTag 与 ARToolkitPlus-BCH 的误检率；光线追踪仿真下有效测距稳定至 50 m（ARToolkitPlus 在约 25 m 检出率降至 50% 以下）；Java 实现 VGA 30 fps（Core2 2.6 GHz）。

### 2. AprilTag 2（Wang & Olson, IROS 2016）

- **取舍核心：** 实证发现用户极少使用纠错 >2 bit 或边框部分遮挡的候选结果，因此以「放弃部分遮挡边框恢复」换取速度提升、小标签召回提升及更低误检率。
- **关键改动：**
  1. **自适应二值化：** `4×4` 分块在 `3×3` 邻域内取极值，以 `(max+min)/2` 切分，跳过低对比度灰区。
  2. **连续边界分割（Continuous Boundary Segmentation）：** 用并查集标定黑白连通域 ID，以 `(r0, r1)` 成对 ID 作为哈希键聚类交界像素，解决极小/远距标签中仅 1 像素白边导致相邻区域错误粘连的问题。
  3. **高效 Quad 拟合与快速解码：** 按重心角排序后利用一二阶矩前缀和在 $O(n)$ 时间以 PCA 求角点；限制纠错上限为 `<=2 bit` 后预构建哈希表，解码由 $O(|C|)$ 降为 $O(1)$；可选梯度法线加权亚像素边缘精化提升位姿精度。
- **实验：** LabelMe 421,049 张图像（`36h11`，允许 2 bit 纠错）候选四边形由 51,075,971 降至 13,623,725，误检由 145 降至 6（误检率 0.000284% -> 0.000044%）；Xeon 2.5 GHz 单核耗时由 0.374 us/px 降至 0.254 us/px（VGA ~78 ms），`decimate=2` 下仅 0.072 us/px（VGA ~22 ms）且不损失定位精度。

### 3. AprilTag 3：Flexible Layouts（Krogius, Haggenmiller & Olson, IROS 2019）

- **灵活布局（`w/b/d/x` 字符串）：** 突破传统方形外黑边套内数据位结构，支持把数据位外移至边框外层形成 **uramaki** 布局（例如 `9×9` 的 `tagStandard41h12` 提供 2115 个 ID 且 Hamming 距离 12，反超 `10×10` `maki` `tag36h11` 的 587 个 ID；`tagStandard52h13` 含 48714 个 ID），还支持圆形标签（`21h7`、`49h12`）及中心挖空嵌套的递归标签（实测 90.2 cm 双重嵌套标靶可在 640×480 下覆盖 `0.08–16.15 ± 0.03 m` 无人机降落范围）。
- **Ising 能量复杂度度量：** 在 LabelMe 207,920 张图片上对比矩形复杂度、连通域数与 Ising 边界总长度能量，Ising 能量在 `41h12` 与 `21h7` 上取得最低误检表现。
- **AprilTag 3 检测加速：**
  1. 点采样降采样（Point Sampling Decimation）替代盒式滤波以保留边缘清晰度；
  2. 优化并查集行扫描条件跳过冗余 union，并提前剪枝过小连通域；
  3. 用象限+斜率替代 `atan2` 排序，以协方差最小特征值直接评估直线拟合残差；
  4. 固定 $H_{33}=1$ 将 9×9 SVD 单应矩阵求解化简为 8×8 高斯消元；
  5. 双线性插值采样 bit 中心并在解码前叠加 $3\times 3$ 拉普拉斯锐化（$K = I + 0.25 L$）以提升极小标签检出。
- **实验：** 在 i7-7600U 2.80 GHz 上通过降采样/最小尺寸参数扫描对比 Recall-FPS 曲线，AprilTag 3 整体优于 AprilTag 2 与 ArUco 3（`DM_FAST`）。
