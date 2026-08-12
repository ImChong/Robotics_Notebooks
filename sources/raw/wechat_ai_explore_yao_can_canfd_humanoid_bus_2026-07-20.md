---
title: 一文读懂CAN与CAN FD：从车载诞生到人形机器人底层总线
author: AI探索Yao
date: "2026-07-20 18:57:06"
source: "https://mp.weixin.qq.com/s/UvjlH1bCsZwNHC2_z12cBg"
---

# 一文读懂CAN与CAN FD：从车载诞生到人形机器人底层总线

![Image](https://mmbiz.qpic.cn/mmbiz_png/sWFCiao2AAVBSJQVycIv8v82wGOT2Pk6SgWqDCBCcm2cRt5OUzOJ9q6M0EEVtR0pibCyOXYHNoaO25oBP41QQ1yrM0FstygBTGKoe3OaMGHoQ/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=0)

## 前言

CAN总线诞生初衷是解决汽车内部电子设备布线臃肿、通讯不稳定问题，经过数十年迭代衍生出升级版CAN FD。如今这套总线早已不止用于汽车，人形机器人灵巧手、分布式轻载关节、全身传感器都在大规模搭载。本文完整梳理技术发展史、协议帧结构、传输参数，同时结合人形机器人落地场景讲清选型逻辑。

## 一、CAN / CAN FD 完整发展历史（起源车载通信）

### 1. 经典CAN总线（CAN 2.0）诞生

- **1983年**：德国博世Bosch启动研发，核心目标：用一根双线替代整车ECU点对点串口线束，减重、降本、抗电机电磁干扰。
- **1986年**：汉诺威车展正式对外发布CAN技术，主打**多设备共线、差分抗干扰、硬件优先级仲裁**三大核心能力。
- **1993年**：ISO发布国际标准ISO 11898，拆分三层规范：

- ISO 11898-1：数据链路层协议（帧、校验、仲裁规则）
- ISO 11898-2：高速CAN物理层，最高1Mbps，动力底盘使用
- ISO 11898-3：低速容错CAN，最高125kbps，车身灯光、门窗使用

- **分支规范**

- CAN 2.0A：标准帧，11位ID，小型车身设备
- CAN 2.0B：扩展帧，29位ID，复杂整车、多节点网络，向下兼容2.0A

### 2. CAN FD 升级背景（Flexible Data-Rate 灵活速率CAN）

随着智能汽车ADAS传感器增多、人形机器人多关节数据爆发，经典CAN暴露两大硬伤：单帧仅8字节、全程最高1Mbps带宽，多传感器同时上报极易总线拥堵。

- **2011年**：博世启动CAN FD研发，保留CAN全部成熟机制，仅扩容数据长度、支持变速传输。
- **2012年**：发布CAN FD白皮书，确定双波特率、64字节最大载荷核心方案。
- **2015年**：正式纳入ISO 11898-1:2015国际标准，硬件完全向下兼容传统CAN设备。

## 二、经典CAN 2.0 完整技术规范

### 1. 传输速率参数

- 高速CAN（动力/底盘）：125kbps ~ 1Mbps，线缆越短速率越高；1Mbps仅支持40米内布线
- 低速容错CAN（车身外设）：125kbps，最长1000米
- 硬性限制：整帧全程同一速率，无分段变速

### 2. CAN 数据帧完整字段（7大段）

![Image](https://mmbiz.qpic.cn/mmbiz_png/sWFCiao2AAVAS7ic9KosmwOFR4WCPwHE1WbQWh4T6ib3EOX4wvPUR3aUtkcTOeCgpb9XOaoZ5iaskbjRScqu48yEJCBfuzkr7D7guk9AVjPZF8Y/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=1)

### 标准数据帧由7个固定位场依次组成：

- SOF帧起始：1位显性电平，标志报文开始
- 仲裁场：11位/29位ID + RTR远程请求位；ID数字越小，报文优先级越高（刹车、急停指令可设最高优先级）
- 控制场：6位，标识后续数据字节长度DLC（0~8字节）
- 数据场：0~8字节有效负载，存储电机指令、传感器数值
- CRC校验场：15位循环校验，检测传输干扰导致的数据错乱
- ACK应答场：接收节点正确接收后回传应答位
- EOF帧结束：7位隐性电平，单条报文终止

### 3. 经典CAN核心短板

- 单帧最大仅8字节，多传感器批量数据需要频繁分包，占用总线资源
- 整段报文只能单一波特率，无法提速
- 多节点并发时，大量分包报文造成总线负载过高，控制延迟变大

## 三、CAN FD 升级技术规范（对比CAN 2.0）

### 1. 传输速率核心特性：双波特率机制

- 仲裁段（ID、控制位部分）：速率≤1Mbps，和传统CAN保持一致，保证新旧设备兼容、总线仲裁逻辑不变
- 数据段（有效载荷区域）：硬件支持最高5~8Mbps，大幅提升大数据传输速度

### 2. CAN FD 数据帧字段改动

![Image](https://mmbiz.qpic.cn/mmbiz_png/sWFCiao2AAVAl1vRJuQevvfbBT1En0FFe8Tf2czdyKLFMicQmb8GwOIcckG7aobknFHdUtJh9Z3I63QvyY9F8K53wnqOZXslrdUaFOJNx8MRw/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=2)

### 整体帧结构和CAN高度相似，仅两处关键升级：

- 控制场新增**BRS速率切换位**：置1代表进入数据段后切换高速传输
- 数据场容量大幅扩容：DLC编码支持0~64字节，单次可打包多路关节温度、力矩、编码器数据
- CRC校验升级：长帧使用26位CRC，相比CAN 15位校验，长数据包抗错误能力更强
- 取消远程帧，仅保留数据帧、错误帧、超载帧

### 3. CAN 2.0 VS CAN FD 核心参数对照表

| 参数 | CAN 2.0 | CAN FD |
| --- | --- | --- |
| 单帧最大有效数据 | 8字节 | 64字节 |
| 传输速率 | 全程最高1Mbps | 仲裁1Mbps，数据段最高8Mbps |
| CRC校验长度 | 15bit | 17/26bit自适应 |
| 硬件兼容性 | CAN FD设备可收CAN报文，反之不行 | 向下兼容传统CAN |
| 多传感器传输效率 | 低，频繁分包 | 高，单帧打包多路数据 |

## 四、CAN / CAN FD 在人形机器人中的落地场景

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/sWFCiao2AAVC4f5yVYibeHzGlovf2qM3jURibU9E2lYaxnSRlVZzFAsz6scQm4QmSx1bUv3NZJvJqVeNISqtd844He5N5avW9kpFzo3viaOmI5o/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=3)

目前全尺寸量产人形机器人统一采用**分层通信架构**：主干EtherCAT以太网负责髋膝大功率关节、视觉图像；多路CAN FD作为末端分支总线，负责小型执行单元。

### 1. 五指灵巧手（使用最多）

- 硬件：单根CAN FD总线串联5根手指微型伺服电机、指尖压力传感器、限位开关
- 传输数据：抓取力矩、手指角度、指尖压力、电机温度
- 选型理由：手部内部空间狭小，仅需2根双绞线布线，无需交换机；单帧64字节可一次性上报5个手指全部传感数据，不用频繁分包

### 2. 全身轻量化小关节

适用：手腕、颈部俯仰旋转、脚踝微调、小臂轻载关节

- 一条CAN FD总线可串联8~12个小型关节模组
- 传输：位置环指令、力矩反馈、电机过热故障、抱闸开关信号
- 区分：大腿、髋关节大功率主关节不使用CAN FD，走EtherCAT，需要微秒级多轴同步

### 3. 全身分布式传感器阵列

- 足底多维力传感器：机器人平衡、行走姿态反馈
- 颈部/手腕分布式IMU姿态传感器
- 机身碰撞检测、整机温度采集模块
  全部依靠CAN FD周期性上传传感数值。

### 4. 整机电源与安全回路

- 锂电池BMS管理：电芯电压、温度、绝缘检测、充放电保护信号
- 整机急停STO安全信号：最高优先级报文，电机紧急断电指令
- 整机固件OTA刷写：64字节大包大幅缩短程序烧录时间

### 5. 诊断与整机状态交互

中控主板通过CAN FD网关，采集所有关节故障码、运行日志，实现整机故障自检。

## 五、人形机器人为什么优先选CAN FD，不用串口/普通CAN/以太网？

### 1. 对比TTL串口

- 串口只能一对一通讯，10个电机就要10组线路，机身线束臃肿；CAN FD一条线串联十几个设备
- 串口无硬件冲突仲裁，多设备同时发数据直接乱码；CAN硬件自动按ID优先级排队，安全指令优先传输
- TTL单端信号抗干扰极差，电机启停电磁干扰直接丢包；CAN差分双线传输，强电磁环境稳定通讯
- 串口无硬件故障隔离，一个外设短路整条链路瘫痪；CAN FD单个节点损坏仅自身离线，其余关节正常工作

### 2. 对比老式CAN 2.0

- CAN单帧仅8字节，多路传感器数据需要反复分包，总线拥堵，关节控制延迟高
- CAN FD 64字节大包+数据段高速率，同等数据量总线负载降低60%以上，控制频率可稳定维持1kHz以上

### 3. 对比EtherCAT工业以太网

- 成本差距：通用MCU芯片原生自带CAN FD外设，收发器成本极低；EtherCAT需要专用ESC芯片，硬件成本翻倍
- 布线结构：CAN FD总线一串到底，适配手指、脚踝狭小腔体；以太网必须星型/菊花链独立布线，末端空间放不下
- 带宽匹配：末端仅传输控制指令、传感器数值，无图像/点云大数据，CAN FD带宽完全够用，以太网性能过剩

## 六、文章总结

![Image](https://mmbiz.qpic.cn/mmbiz_png/sWFCiao2AAVD7S1ibtzZOwCWG1b41ibrZVibrtJsMz6jW1YZpX8zQJibGVMcssBW7eoHVgibNhhoztllPNcsZtLzDfndG88kgzMCbo0LiavOPichnjU/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=4)

1. ## CAN起源于80年代车载电子，核心解决多设备共线可靠通讯；CAN FD是2015年标准化升级版本，解决传统CAN数据量、速率瓶颈，物理层硬件完全兼容。
2. 协议核心差异：CAN最大8字节、单速率；CAN FD支持64字节载荷、双段变速传输，校验机制更强。
3. 人形机器人通信分层逻辑：主干以太网处理大功率关节与视觉大数据；CAN FD作为末端总线，承载灵巧手、轻载关节、分布式传感器、电源安全系统，兼顾低成本、小布线、强抗干扰三大优势，是当下量产人形机器人的标配底层通信方案。
