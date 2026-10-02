---
title: 具身智能入门④：不碰硬件，把云端训出的 Microduck ONNX 塞进 Rust 运行时
author: 智践行
date: "2026-09-28 07:55:00"
source: "https://mp.weixin.qq.com/s?__biz=Mzk2NDU0MzA3OA==&mid=2247492524&idx=1&sn=03dfcab89f09e598d946cb073c69ae5e"
---

# 具身智能入门④：不碰硬件，把云端训出的 Microduck ONNX 塞进 Rust 运行时

上一篇（**[读懂 PPO 训练日志](https://mp.weixin.qq.com/s?__biz=Mzk2NDU0MzA3OA==&mid=2247492496&idx=1&sn=767310ed64bd0345a7dc42381f9fe597&scene=21#wechat_redirect)**）你亲手把策略训了出来、导出成 ONNX，也确认了它在推理链路上能跑通。但那一步的推理，是用 Python 脚本跑的，离部署侧的运行时还差着最后一段。

**策略从 Python 脚本换到 Rust 运行时里跑，这一换我们一直还没碰**。

这篇补的就是这一段：不买硬件、不接舵机，把云端训好的 ONNX 模型加载进 Microduck 的 Rust 运行时，喂一步观测、吐一组舵机指令，确认从仿真到边缘的推理链路是通的。

先把结论放前面，给你吃颗定心丸：**这一步不需要真机，也不需要重新训练。你手里**`Mjlab-Velocity-Flat-MicroDuck`**云端导出的 ONNX 模型，直接就能用**。

## 01 仿真侧闭合了，不等于部署侧通了

先花一分钟把地图在脑子里摆正，后面每一步才知道自己在哪。

通过之前的文章你建立了认知、看懂了数据流，亲手把策略训了出来、导出成 ONNX。链路在**仿真侧**已经闭合了。

但仿真侧闭合，不等于部署侧通了。

### 赛车游戏和真赛道

你在 Python 里能跑推理，和这份策略真的能被 Rust 固件加载、在控制环里稳定吐出动作，中间还隔着**一段真实的距离**。

先打个比方。你在电脑的赛车游戏里把赛道跑通了，这叫仿真侧闭合；现在要把这辆车开上真实赛道，这叫部署到机器人。游戏里能跑，和真车真的能稳定跑，中间隔着一截真实的距离，因为游戏帮你把路况、油门、轮胎都模拟好了，真车却得自己面对油够不够、轮子装没装对。

### 差距不在策略，在环境

两边用的是同一份 ONNX 模型、同一个推理内核 ONNX Runtime，区别在外面的壳：Python 那层壳是 Python 包，Rust 那层壳是机器人上的程序。好比同一个视频文件，电脑和手机都能放，画面一样。所以能在 Python 里跑推理这件事本身，证明的是**策略没问题**。

**但是，问题恰恰出在壳上**：

- **仿真环境是被照顾好的**：推理引擎安装时自动带好，观测由仿真器按正确格式生成并保证正确，根本不用操心硬件。
- **Rust 固件啥都得自己来**：推理引擎得自己去系统里找到并加载，观测得自己从真实传感器拼出来，还得在控制环里自己一步步地驱动策略算出动作。

### 三个坑，只有上了 Rust 才碰得到

- **动态库在不在**：推理引擎不会被自动带上，得运行时自己去系统里找。
- **观测维度对不对**：策略只认一列固定长度的数字，少一个多一个都当场报错。
- **有没有 NaN 这类数学黑洞**：NaN（Not a Number，算不出来的数）混进状态里，推理就失效。

### 这一篇做什么：mock 调试

mock 就是**拿假数据当替身**，不接真机、也不开仿真，拿一份预先录好的观测轨迹，在 Rust 端离线回放，专门验证模型能不能被加载起来、把加载时的校验全部触发。

真机本该自己读传感器、自己下舵机指令，这里用录好的轨迹顶替掉传感器，让策略以为自己在真机上跑。

![全链路架构图：Sim 训练 → output.onnx → Rust 运行时（duck-control） → 14 路动作，高亮本篇覆盖的运行时到动作段](https://mmbiz.qpic.cn/sz_mmbiz_png/pibBMgfstibfm4s6wOxwNiboOoMf91X7dMjy37LaKu57rBpyR4OEYLuMbdtUbn67Nc8nMeKunl65yYtkAl2GDqgZveozOW0t3A0weiaFa8qOnWE/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=0)全链路架构图：Sim 训练 → output.onnx → Rust 运行时（duck-control） → 14 路动作，高亮本篇覆盖的运行时到动作段

## 02 准备

四件事按顺序做。前两件你大概率已经备好了，这里顺手核对；后两件是这篇新增的。**先定好场地：这一节到第 04 节，命令都在你自己的电脑上敲，云端实例暂时用不上**，等第 05 节录观测数据时，再回云端取一下数据。

### 2.1 拿到 **ONNX 模型**

这份 ONNX 模型是上一篇 **[读懂 PPO 训练日志导出的 ONNX 模型](https://mp.weixin.qq.com/s?__biz=Mzk2NDU0MzA3OA==&mid=2247492496&idx=1&sn=767310ed64bd0345a7dc42381f9fe597&scene=21#wechat_redirect)**。回顾一下，当时你跑的命令是：

```
uv run scripts/export.py Mjlab-Velocity-Flat-MicroDuck --checkpoint 199
```

导出命令默认得到 `output.onnx`，放在 `microduck_rl/`根目录下。**它现在在云端实例上，这篇要拿它在本地跑，记得用魔搭的文件面板把它下载到你自己的电脑**。

> 红线再强调一次：必须用项目自带脚本导出，别手写 `torch.onnx.export`，会漏掉观测归一化，导出的策略行为直接错乱。这一步的产物是 **读懂 PPO 训练日志**那一篇的交付，这里只管把它拿来用。

### 2.2 克隆运行时仓库

```
git clone https://github.com/pollen-robotics/microduck
cd microduck
```

整套系统由两个仓库组成：`microduck_rl`（Python）负责仿真和训练，`microduck`（Rust）负责机载运行时，左边训练、右边部署，中间的 ONNX 是过桥的船。现在克隆的是部署这半边。这是一个纯 Rust 仓库，没有用任何机器人框架，代码量可控。Rust 把一个仓库有好几个子项目的做法叫 workspace，每个子项目叫一个成员。

相关的成员有两个：

- **`duck-control`**：策略推理核心
- **`robotd`**：50Hz 控制环的守护进程。Hz 是“每秒次数”，50Hz 就是每秒 50 轮、每 20 毫秒一轮，每轮读传感器、跑推理、算舵机目标

进阶还有个可选的 `scripts/duck-sim`，能把真实守护进程对 MuJoCo 身体跑起来做全仿真。

这次只分析 `duck-control`提供的一个离线例子 `policy-rehearsal`。它是 Rust 项目里附带的程序，能直接编译运行，通常用来演示这个库怎么用，不是正式产品代码。

![运行时工程目录树：cargo workspace 里的 duck-control（策略推理）与 robotd（50Hz 控制环）](https://mmbiz.qpic.cn/sz_mmbiz_png/pibBMgfstibflk50C0Zic2Q4K8iaIia6Ce0JzU2p9WUibia8VXfzyQicRIascJxv3ibtkclF16iaLLeXJia0mzmiapUMnGnJwJpJRRFsUrGJ8vfzQuENfP0/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=1)运行时工程目录树：cargo workspace 里的 duck-control（策略推理）与 robotd（50Hz 控制环）

### 2.3 装 Rust 工具链

Rust 的构建工具和包管理器叫 cargo，编译、跑程序、拉依赖都靠它，地位相当于 Python 里的 uv。Mac 还没装的话，一行命令搞定（装完按提示把 cargo 加进 PATH）：

```
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

装好确认一下：

```
cargo --version
```

### 2.4 装 ONNX Runtime 动态库

`duck-control`是用 Rust 写的，它自己不会算 ONNX 模型，得靠一个现成的推理引擎来算，这个引擎就是 ONNX Runtime。

Rust 程序没法直接调用 ONNX Runtime 这个 C/C++ 库，必须有个中间层把两边的接口对上，`ort`就是干这个的，它是连接 Rust 和 ONNX Runtime 的绑定层。Microduck 用的是 `ort`的 `2.0.0-rc.11`版本。

`ort`接上 ONNX Runtime 用的是 **`load-dynamic`** ：编译时**不**把引擎焊进程序（静态链接），而是等到程序**第一次做推理**那一刻，才去系统里把库 `libonnxruntime`找出来、现场加载。

因此**引擎没有随程序打包**，程序跑起来能不能调到它，只看**运行程序的这台机器**上装没装。而这套程序会落在两种芯片上跑，不同架构要的 ONNX Runtime 二进制不是同一份：

- **往机器人那块 ARM 板子交叉编译时**：编译机（一般是你的电脑）不需要准备 ARM 版的 ONNX Runtime。真机跑起来的时候会去加载板子上的 ARM 版 ONNX Runtime
- **在你的电脑上跑推理时**：必须装好**电脑架构**的那一份，x86\\_64 或 Apple 芯片对应的 ONNX Runtime，和上面那份 ARM 库是两回事。

![load-dynamic 两种场景对比：交叉编译不需要 ARM 版库，Mac 跑 mock 必须装 Mac 架构版库](https://mmbiz.qpic.cn/mmbiz_png/pibBMgfstibfkvHicV9icUCMcIRm6UcPZMfrm3oZRO8ic1UqiaWibPmXSLBicoMA6ia1Opl0RiaUsQh5v2nRbINNrwuibNf6mYNjSQfLvTbU5WB5m6jm6Y/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=2)load-dynamic 两种场景对比：交叉编译不需要 ARM 版库，Mac 跑 mock 必须装 Mac 架构版库

还有个容易漏的点：不装库**照样能编译**，也能跑那些不做推理的测试，因为编译器不会去找这个库。所以别以为编译通过了就万事大吉，编译过 ≠ 库装对了。

所以这一步省不掉。下面是我 **Intel 芯片的 Mac 用官方预编译包安装的方法**：

```
# 下载官方预编译包
curl -L -o ~/onnxruntime.tgz https://github.com/microsoft/onnxruntime/releases/download/v1.23.2/onnxruntime-osx-x86_64-1.23.2.tgz

# 解压到 /opt/onnxruntime
sudo mkdir -p /opt/onnxruntime
sudo tar -xzf ~/onnxruntime.tgz -C /opt/onnxruntime
```

解开之后，完整路径应该是 `/opt/onnxruntime/onnxruntime-osx-x86_64-1.23.2/lib/libonnxruntime.dylib`。

装完最好显式指定动态库的位置：

```
export ORT_DYLIB_PATH=/opt/onnxruntime/onnxruntime-osx-x86_64-1.23.2/lib/libonnxruntime.dylib
```

但是，这样设只在当前终端有效。想让它永久生效就把这一行写进 `~/.zshrc`，macOS 的终端每次启动都会读这个文件：

```
# 先看一眼里面有没有已经加过
grep ORT_DYLIB_PATH ~/.zshrc

# 没有的话追加进去（>>是追加，不会覆盖原有内容）
echo 'export ORT_DYLIB_PATH=/opt/onnxruntime/onnxruntime-osx-x86_64-1.23.2/lib/libonnxruntime.dylib' >>~/.zshrc

# 让当前这个终端立刻生效，不用重开
source ~/.zshrc

# 验证：应当打印出上面那串完整路径
echo $ORT_DYLIB_PATH
```

`ort 2.0.0-rc.11`要求 ONNX Runtime 不低于 **1.23**，机器人板子上的系统镜像通常会选更新的稳定版，但底线都是 1.23。

低于这个版本，加载时 `ort`会直接 panic。panic 是 Rust 的说法，意思是程序遇到撑不下去的错误当场中止，并把错误信息打印出来。

装完先确认能编译例子：

```
cargo build -p duck-control --example policy-rehearsal
```

能编译过，说明工具链和依赖都就位了。接下来加载你的 ONNX 模型。

## 03 把 ONNX 加载进 Rust

加载和推理，一条命令一次做完。进到克隆好的 `microduck`目录，把路径换成你的 `output.onnx`，跑：

```
cargo run --release -p duck-control --example policy-rehearsal -- output.onnx
```

终端会打印一段 JSON。**只要没报错、`actions`****里是一串正常的数，你的 ONNX 模型就被 Rust 运行时加载起来、跑出动作了**。

这个例子的文件头有一行原话，先记住它：**Offline inference only: never opens a motor bus**。它不开电机总线，纯离线推理，所以你在 Mac 上跑它，安全，不会动任何硬件。

`policy-rehearsal`这条命令其实连着做了两件事：先把模型加载进来，再拿观测一步步推理。本节把加载这一步拆开讲清楚。

### 3.1 加载做了什么：把 ONNX 变成可以调用的对象

官方离线例子 `policy-rehearsal`的源码地址是 <https://github.com/pollen-robotics/microduck/blob/main/duck-control/examples/policy-rehearsal.rs>，用浏览器打开就能看到全文，其中负责加载 ONNX 的就是下面这一段：

```
// 第一个参数：每个模型的文件在哪，walk 必填，其余留空
// 第二个参数：站立阈值 0.05，速度指令的模低于它时切到站立模型
let mut policy = Policy::load(
    &PolicyPaths { walk: path, ..Default::default() },
    0.05,
)?;
```

`Policy::load`把 `output.onnx`读进来，逐项校验，返回一个可以反复调用的 `policy`对象。**注意它返回的是 policy 对象本身，不是动作**，动作要等之后拿一步观测去调 `policy`才算出来。它一共收两个参数：

- **第一个参数****`PolicyPaths`**：告诉它每个模型的文件在哪。`walk`是走路模型，必填，就是你传进来的 `output.onnx`，其它的`stand`、`sitstand`这些别的姿态模型可填可不填，所以末尾的 `..Default::default()`意思是其余全部留空，所以这次加载进来的只有走路这一个模型。
- **第二个参数 0.05**：站立阈值，速度指令的模低于它时切到站立模型。这次没有加载站立模型，它暂时不起作用，用默认值即可。

加载完之后，无论内部多复杂，这个 policy 对运行时来说就是一个函数：观测进、动作出，观测就是机器人“看到”的身体状态，动作就是要下给舵机的指令。

- **输入**`obs`：形状 `[1, 61]`，一步 61 维观测
- **输出**`actions`：形状 `[1, 14]`，14 路舵机目标

61 维观测里装的是：陀螺仪（3）、投影重力（3）、关节位置相对默认姿态（14）、关节速度（14）、上一步动作（14）、以及 13 维命令（前后、左右、转向速度加上头/身体姿态指令）。机器人全身是 15 个舵机，舵机就是能精确转到某个角度的小电机，每个舵机管一个关节；14 路动作对应除鸭嘴外的那 14 个。

![加载与推理两步走：output.onnx 经 Policy::load 加载并校验，得到 policy 对象；之后每来一步 61 维观测，推理才输出 14 路动作](https://mmbiz.qpic.cn/sz_mmbiz_png/pibBMgfstibfl9j1KZAgMRVpibUGNWg7f8OLBbKvM2JC02bRQgmS4icxrhfoEibcuaEobWnBVRmiasicN6tLJOXIYsWX8PIia4hrtYH1ckvIpCcGFMo/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=3)加载与推理两步走：output.onnx 经 Policy::load 加载并校验，得到 policy 对象；之后每来一步 61 维观测，推理才输出 14 路动作

### 3.2 两种策略：带不带记忆

模型有两种版本，区别只有一件事：它记不记得上一轮发生过什么。

- **feed-forward 版**：不带记忆。每次推理只看当前这一步，`obs`进、`actions`出，这一步算完什么都不留，下一步重新看。你在 **免费云 GPU 跑通 PPO 训练**里训出来的默认是这一版。
- **recurrent（LSTM）版**：带记忆。除了这一步观测，还要把上一轮留下的记忆一起喂进去，算完再把更新过的记忆吐出来，留给下一轮接着用。

recurrent 版多出来的四个量，合起来就是这段记忆，名字拆开看很好认：

- `h_in`和 `c_in`：这次推理**喂进去**的上一次记忆
- `h_out`和 `c_out`：这次推理**吐出来**、要交给下一次的记忆
- `h`记得近，管最近这几轮；`c`记得远，管更久之前一路攒下来的东西。`in`就是进，`out`就是出

带上记忆之后，每一轮推理就不再孤立：这一轮吐出的 `h_out`/ `c_out`，正是下一轮喂进去的 `h_in`/ `c_in`，一轮一轮往下传。

每次推理跑完，Rust 运行时会自动把新的 `h`和 `c`抄回下一轮要用的位置，源码里对应的就是两行 `copy_from_slice`（在 `duck-control/src/policy.rs`里搜这两个词）。调用时只管喂观测、取动作。

这一项和你的关系是：你自己训的那版是 feed-forward，这四个记忆口子用不上。

### 3.3 加载时校验，而不是推理时才崩

这是 Microduck 运行时设计上最值得记住的一点，源码注释原话是：**Everything is validated at load, not at inference**。

意思是：观测宽度不对、动作数量不对、ONNX Runtime 没装，这些错误**在加载阶段、机器人还站着不动的时候就报出来**，而不是走到第 60 个控制周期、鸭子正迈步时才炸。

对真机来说这生死攸关。加载失败会被变成**保持姿势、上报异常**，升级系统看到异常就回滚，而不是留一只走不了的鸭子。

你在**把 ONNX 加载进 Rust** 命令时，这套校验已经替你跑过一遍了：

- `obs`输入必须是 `[1 或动态批, 61]`
- `actions`输出必须是 `[1 或动态批, 14]`，且每个值都有限（不能是 NaN/Inf）
- 如果是 recurrent 版， `h_in`/ `c_in`/ `h_out`/ `c_out`的形状必须合规
- ONNX Runtime 动态库必须能 `dlopen`成功

任何一项不符， `Policy::load`直接拒绝，连第一步推理都不会发生。

### 3.4 加载之后：推理这一步

加载完拿到 `policy`对象，接下来就是反复调用它：每步喂 61 维观测，就吐出一组 14 路动作。例子默认跑 1000 步，所以你会拿到 1000 组动作。

## 04 读懂 mock 调试的输入与输出

下面是这份 ONNX 模型跑 1000 步基准得出的只有 JSON 结果，没有 3D 窗口。

```
{
  "steps": 1000,
  "latency_ms": { "p50": 0.026, "p95": 0.034, "p99": 0.056, "max": 0.124 },
  "over_20_ms": 0,
  "actions": [[一组 14 个浮点数], ... 一共 1000 组],
  "scope": "actor inference only; excludes sensor and motor I/O"
}
```

其中 **`actions`****不是一组 14 个数，而是 1000 组**。跑了多少步，就有多少组动作，每组 14 个。因为默认基准是 1000 步，所以它是一千组。

几个字段挨个说：

- **`latency_ms`**是推理延迟，单位毫秒，给出 p50 / p95 / p99 / max 四个值
- 这四个都是分位值：把这 1000 步的耗时从快到慢排队，p50 是排在正中间那一步的耗时，p95 是只有 5% 的步数比它慢的那个值，p99 是只有 1% 比它慢，max 就是最慢的一步
- 实测里最慢一步是 0.124 毫秒，中位数 0.026 毫秒，都不到 20 毫秒控制周期的百分之一，说明在 50Hz 环里跑推理时间绰绰有余
- **`over_20_ms`**是超过 20 毫秒的步数。控制周期是 20 毫秒，这一项如果是 0，说明没有一步会拖垮控制环。
- **`actions`**就是那 14 路舵机目标，正是 `robotd`在真机上每个周期会写给舵机的东西。
- **`scope`**这句话很重要：`actor inference only; excludes sensor and motor I/O`。它在声明：这次只测了**给观测、出动作**这一小段，不含传感器读取和电机写入。换句话说，**它验证的是推理链路通不通，不是鸭子走得好不好看**。

> 注意它和 **读懂 PPO 训练日志**里 `infer_policy.py`的体验一模一样：都是无头、只有日志、没有会动的画面。

**看到这段 JSON、没有报错、14 路动作都在有限值范围内，从训练到部署的推理链路，你就在 Mac 上完整走了一遍**。

## 05 对齐验证：Rust 输出 vs 仿真输出

光自己能跑还不够。要证明这份 ONNX 在 Rust 里跑出来的，和你在仿真里跑出来的，是同一份策略，得做一件事：拿同一段观测喂给 Rust 推理，再把它算出的 14 路动作，和云上仿真那次 Python 端算出的动作对一遍。

### 5.1 同一段观测，两次结果做对比

准备一步的 61 维观测轨迹。轨迹是一段 JSON 数组，每条是一步观测加一个重置标志：

```
[
  {"obs": [61 个浮点数], "reset": false},
  {"obs": [61 个浮点数], "reset": false}
]
```

这 61 个数要从仿真里真跑一次取出来，不能凭空填。而且得在训练用的那个仿真里取，地面、摩擦、传感器读数和训练时一致，录出来的观测才对得上策略见过的分布，所以这一步要回到 **[免费云 GPU 跑通 PPO 训练](https://mp.weixin.qq.com/s?__biz=Mzk2NDU0MzA3OA==&mid=2247492482&idx=1&sn=f2f318a943b8f8ae45a48a6907cabda6&scene=21#wechat_redirect)**那篇用过的实例，把它开机，登上终端，取完数据再回到 Mac：

`infer_policy.py`默认的屏幕打印只有速度行，不打印观测，要额外加 `--save-csv`才会把每一步的观测写进文件：

```
# 先登上云 GPU 实例的终端，下面几行都在那里执行
# 安装 xvfb
sudo apt-get update && sudo apt-get install -y xvfb

# 配置实例环境
source /mnt/workspace/setup_env.sh

cd microduck_rl
xvfb-run -a uv run scripts/infer_policy.py --walking output.onnx \
  --new-cmd-obs --save-csv obs.csv
```

跑起来后终端照样只有那些速度行，别以为参数没生效。让它走几秒，觉得够了按 `Ctrl+C`。**数据只在退出这一刻才写入磁盘**，没退出之前 `obs.csv`一个字节都还没有。这里有个坑：无头模式下按 Q 键退出是失效的，脚本发现键盘没有接在终端上就不读键盘了，所以只能 Ctrl+C。

退出后当前目录多出一个 `obs.csv`。第一行是表头，往后每一行对应一步，列依次是 `obs_0`到 `obs_60`这 61 个观测值，接着是 `action_0`到 `action_13`这 14 路动作。

我们需要先把第 100 步这一行取出来转成 `trace.json`。转之前有个坑要注意：`obs_34`到 `obs_47`这 14 个位置，装的是喂给模型的上一步动作。有人会问：不是说 feed-forward 无记忆吗，观测里怎么还带着上一步动作？无记忆说的是模型内部不存历史，这里的上一步动作是脚本在喂观测前抄进输入里的上一步动作，属于观测本身，模型只是把它当普通输入读掉，读完也不留。另外， CSV 里写的是**本步**的动作，和同一行的 `action_0`到 `action_13`完全相同。原因是脚本先推理、后记录，等记下来的时候，上一步的旧值已经被本步动作盖掉了。所以要还原真正喂给模型的那 14 个数，得回**上一行**取，也就是取第 99 步的数据。

![obs.csv 一行的列布局：61 个观测值分成 8 段，末尾 14 列是动作](https://mmbiz.qpic.cn/mmbiz_png/pibBMgfstibfnFLGrDQiaXdbODxciaWTicibT7PB8HFHNhTGExwMHplcibncZvLSmgwgmKVbjLHibq4MWmCnS1IjyJRcIjibYq3ufKmhm0SWS8r8bUBA/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=4)obs.csv 一行的列布局：61 个观测值分成 8 段，末尾 14 列是动作

`step`这一列从 1 数起，Python 的下标从 0 数起，所以第 100 步落在 `rows[99]`：

```
python -c "
import csv, json
rows = list(csv.DictReader(open('obs.csv')))
n = 100                                       # 取第 100 步
obs = [float(rows[n-1]['obs_%d' % i]) for i in range(61)]
prev = [float(rows[n-2]['action_%d' % i]) for i in range(14)]
obs[34:48] = prev                             # 这 14 位换成上一步的动作
json.dump([{'obs': obs, 'reset': False}], open('trace.json', 'w'))
print('已写入 trace.json，步数 1，观测维度', len(obs))
print('前三个（机体角速度）', obs[:3])
print('上一步动作前三个', obs[34:37])
"
```

它会直接打印三行，我这次实跑的输出是：

```
已写入 trace.json，步数 1，观测维度 61
前三个（机体角速度） [-0.0023186603, 0.0072314986, -0.0011854586]
上一步动作前三个 [-0.05491654, -0.08619693, 0.029419124]
```

最后一行那三个数，就是 `obs.csv`里第 99 步的 `action_0`到 `action_2`，说明往前挪一行这一步生效了。

而取中间第100步则是因为开头几帧的关节角度还是零，那一刻的观测不能代表策略真正在跑的样子。

生成的 `trace.json`完整内容如下，第一行是 `obs_0`到 `obs_33`，第二行开头的 14 个数，就是往前挪一行换进来的第 99 步动作：

```
[{"obs": [-0.0023186603,0.0072314986,-0.0011854586,0.021222824,0.011456485,-0.9997091,-0.034153428,0.009229675,-0.05697751,0.002474075,0.04117173,-0.12453349,0.09026563,0.0058162524,0.040678527,0.098063536,0.0016723797,0.024947226,0.04395597,0.043496907,-0.00031522478,0.0001244857,-0.0021163402,-0.0008777364,-0.00013418205,0.0005124874,-0.000135668,-0.0003088133,-0.00037267632,0.00030032493,-0.00021681705,-0.00038042193,0.0052064476,-0.0003144523,
-0.05491654,-0.08619693,0.029419124,-0.27395037,0.16016261,-0.061742686,-0.0036816676,-0.01629213,0.010160029,0.11800091,0.0669744,0.060323928,0.178839,0.07320875,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0], "reset": false}]
```

`trace.json`生成在云实例上，我的 Rust 端跑在 Mac 上，所以先把它下载下来。回到 Mac 跑 Rust 端：

```
# Rust 端：回到 Mac，policy-rehearsal 读这段轨迹，输出 actions
cargo run --release -p duck-control --example policy-rehearsal -- output.onnx trace.json
```

跑完终端会吐一行 JSON：

```
{"actions":[[-0.05548464506864548,-0.08708086609840393,0.02850097417831421,-0.2741580009460449,0.15970322489738464,-0.062065206468105316,-0.0038030301220715046,-0.016274675726890564,0.011035226285457611,0.11793065071105957,0.06616401672363281,0.06077351048588753,0.17698998749256134,0.07526372373104095]],"latency_ms":{"max":0.052487,"p50":0.052487,"p95":0.052487,"p99":0.052487},"over_20_ms":0,"scope":"actor inference only; excludes sensor and motor I/O","steps":1}
```

接着就可以拿这个 Rust 端跑出来的 14 路动作，跟 `obs.csv`里那一行 Python 端已经算好的动作做对比。

![对齐对比：同一段观测下，云上 infer\_policy.py 算出的动作与 Mac 上 policy-rehearsal 算出的动作并排](https://mmbiz.qpic.cn/sz_mmbiz_png/pibBMgfstibflNmxtHtcKic3ica1jnAjEIJ6tQ5oB6WwiaInvV3EAMRSDlt5ycDibVOZCeOhg4mnvRrAULspsUpEg7J3f9MhFFuXBuic39QJicxCdMo/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=5)对齐对比：同一段观测下，云上 infer\\_policy.py 算出的动作与 Mac 上 policy-rehearsal 算出的动作并排

`policy-rehearsal`的 `actions`数组，和 `obs.csv`第 100 行里 Python 端已经算好的 14 路动作，应当在浮点误差范围内逐位一致。我实跑出来的对比如下：

| 路 | 云上 onnxruntime | Mac Rust 端 | 差的绝对值 |
| --- | --- | --- | --- |
| 0 | -0.055484645 | -0.055484645 | 6.9e-11 |
| 1 | -0.087080866 | -0.087080866 | 9.8e-11 |
| 2 | 0.028500974 | 0.028500974 | 1.8e-10 |
| 3 | -0.274158000 | -0.274158001 | 9.5e-10 |
| 4 | 0.159703220 | 0.159703225 | 4.9e-9 |
| 5 | -0.062065206 | -0.062065206 | 4.7e-10 |
| 6 | -0.003803030 | -0.003803030 | 2.2e-11 |
| 7 | -0.016274676 | -0.016274676 | 2.7e-10 |
| 8 | 0.011035226 | 0.011035226 | 2.9e-10 |
| 9 | 0.117930650 | 0.117930651 | 7.1e-10 |
| 10 | 0.066164020 | 0.066164017 | 3.3e-9 |
| 11 | 0.060773510 | 0.060773510 | 4.9e-10 |
| 12 | 0.176989990 | 0.176989987 | 2.5e-9 |
| 13 | 0.075263724 | 0.075263724 | 2.7e-10 |

**最大差 4.9e-9，平均 1.0e-9**，而这些动作本身的量级是 0.27，差了八位有效数字。这就是浮点精度本身的误差，可以认定两端算的是同一件事。

需要注意的是，要是忘了把那 14 位往前挪一行，同样的对比最大差会变成 1.08e-3。看着也差不多，但那已经不是精度误差，而是喂进去的观测本身不一样了。

### 5.2 一致意味着什么

两边跑的是同一个 ONNX 模型，只是推理引擎不同：一边是 Rust 的 `ort`/ ONNX Runtime，一边是 Python 的 onnxruntime。Python 那一次是在云上采集数据时跑的，结果留在了 `obs.csv`里。

如果同一段观测下 14 路动作对得上，就说明：

**这份策略在仿真里学到的本事，原封不动地带到了边缘端**。Sim-to-Real 的链路可信，可以放心往真机走了。

## 06 故障排查

`duck-control/tests/fixtures/`下有一套对照模型，文件名基本就是它坏在哪。共 14 个：4 个合法可跑、10 个故意做坏。报错都在\*\*加载阶段（或热身推理）\*\*就抛出，机器人还站着不动时就能看到。

## 怎么操作

复现某个报错，把下面命令里的模型路径换成对应的 fixtures 文件即可：

```
cargo run --release -p duck-control --example policy-rehearsal -- duck-control/tests/fixtures/bad_width.onnx
```

`--`之后才是传给示例程序的参数（模型路径）。

一次看全 10 个坏模型：

```
for m in bad_width bad_batch bad_rank bad_action_count bad_state_shape \
         dynamic_hidden missing_state extra_input wrong_type nan_state; do
  echo "===== $m ====="
  cargo run --release -p duck-control --example policy-rehearsal -- duck-control/tests/fixtures/$m.onnx 2>&1 | head -3
done
```

## 坏模型速查（10 个）

| 文件 | 报错特征（一句话） | 怎么修 |
| --- | --- | --- |
| `bad_width.onnx` | 观测第二维是 60，不是 61 | 按 61 维规范重新导出 |
| `bad_batch.onnx` | 观测批次维是 2 | 导出别写死批次大小 |
| `bad_rank.onnx` | 观测成了 3 维 `[1,1,61]` | 观测应是 2 维 `[1,61]` |
| `bad_action_count.onnx` | 动作输出宽是 -1（动态） | 导出 14 路动作的模型 |
| `bad_state_shape.onnx` | LSTM 四个状态形状对不上 | 导出 `model_api: 2`的 recurrent 模型 |
| `dynamic_hidden.onnx` | `h_in`隐藏维是动态的 | 状态隐藏维用固定正数 |
| `missing_state.onnx` | 图里删掉了 `c_in`输入，图非法 | 用标准导出脚本，别手改图 |
| `extra_input.onnx` | 多了一个不认识的输入 | 用标准导出脚本 |
| `wrong_type.onnx` | 观测是 double 不是 float32 | 导出 float32 模型 |
| `nan_state.onnx` | 热身推理时状态混进 NaN | 重新训练或导出，状态不含 NaN |

## 合法对照（4 个，能正常跑）

`feedforward.onnx`、`lstm.onnx`、`lstm_changed.onnx`、`dynamic_batch.onnx`（批次维动态，受支持）。

## 07 小结与下一步

到这一步，把整条链路数一遍：

**[免费云 GPU 跑通 PPO 训练](https://mp.weixin.qq.com/s?__biz=Mzk2NDU0MzA3OA==&mid=2247492482&idx=1&sn=f2f318a943b8f8ae45a48a6907cabda6&scene=21#wechat_redirect)**，你在云端把策略训了出来

**[读懂 PPO 训练日志](https://mp.weixin.qq.com/s?__biz=Mzk2NDU0MzA3OA==&mid=2247492496&idx=1&sn=767310ed64bd0345a7dc42381f9fe597&scene=21#wechat_redirect)**，你把它导出成 ONNX，并确认这份文件能推理

这篇，你把它加载进 Rust 运行时，在 mock 模式下喂观测、读 14 路动作、确认推理链路通、还和仿真输出对上了

**从仿真训练到边缘部署的推理这一段，你已经亲手走通了。链路通了，不等于鸭子走得稳，**接下来先动训练侧的旋钮把它调稳，再让真机软件栈在仿真身体上跑起来。真机接线和驱动层，留到硬件到位那最后一程。

回顾一下这篇你真正动手做的几件事：

- 装好 ONNX Runtime 动态库（版本不低于 1.23），理解 `load-dynamic`为什么要在 Mac 上装
- 用 `Policy::load`把 `output.onnx`加载进 Rust，知道 61 对 14 的规范在加载时就被校验
- 用 `policy-rehearsal`离线跑推理，只看终端 JSON、不开电机总线，确认延迟远低于 20 毫秒周期
- 把 Rust 输出和 Python 输出对齐，证明策略从仿真原样带到了边缘

下一篇（**Sim-to-Real 调优**）我们就在这条已经走通的推理链路上动手：改奖励函数、开关域随机化，看步态怎么变，把**能跑通**变成**跑得好**。

### 想深挖运行时：这几处够用

上面关于加载和规范只讲了够用的最小版本。真想把它吃透，下面这几处按由近及远排。

最近的就是你刚克隆的那份代码，挑一个啃得动的就行：

**1. 先读你正在跑的这份实现**。 这些加载逻辑和报错信息，都集中在 `duck-control/src/policy.rs`这一个文件里。

`Policy::load`、 `ensure_runtime`、以及 `open`里的形状校验全在这。

<https://github.com/pollen-robotics/microduck/blob/main/duck-control/src/policy.rs>

它的模块注释把**为什么在加载时校验而不是推理时**讲得很透，读代码前先读那段注释。

**2. 想看 61 对 14 规范的权威定义，读****`policy-manifest.md`**：

<https://github.com/pollen-robotics/microduck/blob/main/docs/policy-manifest.md>

`obs_len`、 `action_len`、 `control_hz`、 `model_api`这些字段的含义和拒绝规则都写在这。

这份规范当前是第 2 版，文档里叫 schema 2，schema 就是字段怎么约定的格式版本。recurrent 版要 `model_api: 2`，也是这里定的。

**3. 想搞懂 Rust 侧怎么调 ONNX，读****`ort`****文档**。 Microduck 机载推理实际用的就是它， `load-dynamic`特性的行为在文档里有完整说明：

<https://docs.rs/ort>

**4. 想看 LSTM 版张量规范（隐藏状态生命周期、离线 rehearsal），读****`recurrent-policies.md`**：

<https://github.com/pollen-robotics/microduck/blob/main/docs/recurrent-policies.md>

你 **免费云 GPU 跑通 PPO 训练**训出来的默认是 feed-forward，但官方模型库里有 recurrent 版，提前看懂规范不吃亏。

**5. 想在 Mac 上装对 ONNX Runtime 版本，看官方安装文档**（底线 1.23，板端镜像选更新的稳定版）：

<https://onnxruntime.ai/docs/install/>

喜欢看架构全貌的话， `microduck`仓库根目录的 README 和 `docs/design/`下还有控制环、策略通道等设计文档，顺着链接读即可。

Rust 部署这块有任何卡点，评论区告诉我。下一篇动手调参时，随时回来对照加载和排错两节。



---

相关阅读：[具身智能入门①：0硬件起步，从开源机器鸭Microduck学起](https://mp.weixin.qq.com/s?__biz=Mzk2NDU0MzA3OA==&mid=2247492291&idx=1&sn=853662478d99f4390605c3ec55078d59&scene=21#wechat_redirect)[具身智能入门②：不写一行代码，先把机器鸭Microduck“玩”明白](https://mp.weixin.qq.com/s?__biz=Mzk2NDU0MzA3OA==&mid=2247492360&idx=1&sn=dae6b7ed7716bfb6fdb2c78279cca5f6&scene=21#wechat_redirect)[具身智能入门③（上）：不买显卡也能练，免费云 GPU 跑通 Microduck 的 PPO 训练](https://mp.weixin.qq.com/s?__biz=Mzk2NDU0MzA3OA==&mid=2247492482&idx=1&sn=f2f318a943b8f8ae45a48a6907cabda6&scene=21#wechat_redirect)[具身智能入门③（下）：读懂 PPO 训练产出，导出 ONNX 交给边缘端](https://mp.weixin.qq.com/s?__biz=Mzk2NDU0MzA3OA==&mid=2247492496&idx=1&sn=767310ed64bd0345a7dc42381f9fe597&scene=21#wechat_redirect)
