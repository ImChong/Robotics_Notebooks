# MathWorks 官方 Bode Plot 文档

> 来源归档（official software documentation；核对 MathWorks Control System Toolbox 文档；2026-10-08）

- **bodeplot 函数：** <https://www.mathworks.com/help/control/ref/bodeplot.html>
- **bode 函数：** <https://www.mathworks.com/help/control/ref/dynamicsystem.bode.html>
- **文档说明：** 对动态系统绘制幅值和相位频率响应；可接受传递函数、零极点增益、状态空间及频响数据等模型。
- **连续时间计算：** 在 $s=j\omega$ 上计算频率响应。
- **离散时间计算：** 在单位圆 $z=e^{j\omega T_s}$ 上评估，绘图频率只到 Nyquist 频率 $\pi/T_s$。
- **单位：** 频率输入使用 rad/TimeUnit；可指定频率向量、自动选择或调整频率单位。

## 来源边界

这是工具行为文档，适合核对实现约定和单位；它不是控制理论原始论文。理论历史请参见 Bode 的 1940 年论文、1945 年专著和 MIT 课程讲义。

## 对应 Wiki

- [Bode 图概念节点](../../wiki/concepts/bode-plots.md)
