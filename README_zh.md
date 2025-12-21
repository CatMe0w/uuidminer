# uuidminer

基于 CUDA+OpenCL+Metal 的高性能搜索工具，寻找满足特定前缀的，离线模式 Minecraft 玩家名所对应的 UUID。

在离线模式中，玩家 UUID 由字符串 `OfflinePlayer:玩家名` 的 MD5 哈希确定，再按照 UUID v3 规则设置版本位与变体位。注意到，玩家名空间可被视为一个可枚举的搜索空间。基于这一性质，可对玩家名进行大规模搜索，筛选出 UUID 高位具有特定模式的结果。

## 项目动机与演变

这个项目最初只是一个玩笑：如果把玩家 UUID 当作类似比特币区块哈希的结果，是否可以定义一种“名字难度”，并找出最“困难”的玩家名。

后来，这个想法逐渐变得更加实用：程序性能提升到超越 hashcat，同时，搜索目标不再局限于前导零数量而允许用户指定任意目标 UUID 前缀，例如十六进制字符串 `deadbeef` 或者具有个人意义的日期和数字组合。

现在，uuidminer 已经是通用的离线玩家 UUID 搜索器，不再局限于寻找前导零。

## 功能

程序会枚举合法的 Minecraft 玩家名字符集（不考虑中文玩家名），即 0-9、a-z、A-Z 和 \_，共 63 个字符，搜索长度为 3 到 16 的玩家名。每个候选名字先被映射为对应的离线模式 UUID，再与用户指定的目标前缀进行匹配。

程序支持多 GPU 和多节点并行运行，支持 CUDA、OpenCL 和 Metal 三种后端，可在 Windows、Linux 和 macOS 上运行。

不传递任何参数时，只要 UUID 至少满足 8 个前导零，程序就会输出结果。用户也可以指定任意 8 位十六进制前缀作为搜索目标。

## 编译

需要 CMake >= 3.18、Python 3 以及对应平台的编译器工具链。

默认在 Windows 与 Linux 平台编译 CUDA 与 OpenCL 后端，在 Mac 平台编译 Metal 后端。

```
cmake -B build
cmake --build build --config Release
./build/uuidminer.exe
```

若要指定编译后端，可传递 `-DUSE_CUDA`、`-DUSE_OPENCL` 或 `-DUSE_METAL` 参数。例如，不编译 CUDA 后端：

```
cmake -B build -DUSE_CUDA=OFF
cmake --build build --config Release
```

## 使用方法

```
uuidminer.exe [--target <hex_prefix>] [--backend <backend>] [--node-index N] [--node-count M] [--node-slices S]
```

-   `--target <hex_prefix>`：指定目标 UUID 前缀，长度为 8 个十六进制字符（4 字节）。默认值为 `00000000`，即寻找至少 8 个前导零的结果。
-   `--backend <backend>`：指定使用的后端，可选值为 `cuda`、`opencl` 或 `metal`。默认情况下会自动选择可用的后端。
-   `--node-index`、`--node-count`、`--node-slices`：见“分布式与多节点配置”章节。

## 性能与现状

在当前实现中，单张 RTX 3090 的实测性能约为 44 GH/s，大约相当于 hashcat mode 0 "MD5" 的 **70%**，以及 hashcat mode 20 "md5($salt.$pass)" 的 **126%**。

目前的实现已对这个目标优化到接近最优，由于用户名生成逻辑的存在，它无法达到 hashcat mode 0 纯 MD5 的极限性能。

在这一性能水平下，所有 8 字符长度的搜索结果已经全部挖掘完成。对于 9 字符长度，单张 3090 大约需要连续运行 4 天时间，目前没有计划将其全部跑完。

## 关于前导零阈值的设计取舍

程序以 8 个前导零作为输出阈值，而不是更长的 12 个，即，形如 `00000000-xxxx-xxxx-xxxx-xxxxxxxxxxxx` 而不是 `00000000-0000-xxxx-xxxx-xxxxxxxxxxxx`，原因如下：

在实际搜索中，直到进入 9 字符名字空间才首次出现拥有 12 个前导零的结果。换言之，若单纯为了“两段前导零”这一视觉效果，计算成本将急剧上升，而回报极其有限。

同时，由于 UUID 版本位和变体位的存在，第 13 个字符固定为 `3`，从这里开始，视觉上的“稀有感”已经明显下降。个人认为在 12 位之后继续追求更多前导零并没有明显的展示或研究价值。

最后一个考虑是固定前 8 位的结果足够丰富，大多数目标前缀都能在 6 字符甚至 5 字符空间内找到数十个结果，对大多数用户应该已经足够满足需要。

## 结果分析工具

项目额外提供了一个 Python 脚本 `find_most_difficult.py`，用于对搜索结果进行统计分析。

脚本仅对“前导零”这一经典目标进行分析，可以自行将其修改为分析其他的前缀或模式。

## 分布式与多节点配置

程序支持在多 GPU、多节点环境下运行，所有分布式行为均通过命令行参数控制。

核心参数如下：

-   `--node-count`：整个集群中逻辑节点的总数

-   `--node-index`：当前实例负责的节点编号，从 0 开始

-   `--node-slices`：当前节点额外领取的连续工作分片数量

对于异构算力集群，`--node-slices` 效果拔群。在算力不均衡的环境中，高性能节点可以通过增加 slices 的方式，主动承担更多搜索空间，而无需重新划分整个集群。

### 示例一：均匀算力集群

假设有 4 台性能相近的机器，分别运行：

```
--node-index 0 --node-count 4 --node-slices 1
--node-index 1 --node-count 4 --node-slices 1
--node-index 2 --node-count 4 --node-slices 1
--node-index 3 --node-count 4 --node-slices 1
```

每台机器各自负责四分之一的搜索空间。

### 示例二：异构算力集群

假设有 1 台高算力机器和 2 台低算力机器，若要让高算力节点承担一半工作量：

高算力节点：

```
--node-index 0 --node-count 4 --node-slices 2
```

两台低算力节点：

```
--node-index 2 --node-count 4 --node-slices 1
--node-index 3 --node-count 4 --node-slices 1
```

### 结果输出与 CSV 管道

程序会将命中的结果直接输出到标准输出，格式为：

`playerName,uuid`

若要将结果保存为 CSV 文件：

`uuidminer.exe --target deadbeef > results.csv`

## 已知最困难用户名结果

以下列出目前已经挖掘出的部分“最困难”玩家名结果。

难度定义为 UUID 数值最小。

| Length | Name        | UUID                                   |
| ------ | ----------- | -------------------------------------- |
| 3      | `ign`       | `00006dae-4a2b-3a71-8473-11a2d1553f3f` |
| 4      | `v4K4`      | `00000109-807f-3287-9a54-90aa57e7362a` |
| 5      | `16j63`     | `00000003-a5f8-3217-96d2-c96c3f60529c` |
| 6      | `EhxMb9`    | `00000000-27ce-35d0-b1ca-15b8d6f93fbf` |
| 7      | `tiHSNRY`   | `00000000-0002-3ff2-bda6-bf24a74ffc4e` |
| 8      | `ilAQpWC2`  | `00000000-0003-35b1-9df2-7537067aa4ba` |
| 9      | `00pD0Yk1A` | `00000000-0000-3710-b5b1-7f404c93943b` |

注意：9 字符长度的结果未完全挖掘完成，表格中仅列出目前已知的最小结果。

### 特殊结果

一些知名 [magic number](<https://en.wikipedia.org/wiki/Magic_number_(programming)#Debug_value>) 的结果。

#### deadbeef

| Name     | UUID                                   |
| -------- | -------------------------------------- |
| `CBRVJ`  | `deadbeef-3e36-38de-a786-a2b38b66a1d5` |
| `oHw2J`  | `deadbeef-9ae5-319e-9da3-6a883d7a57c0` |
| `LwXjDy` | `deadbeef-b5a1-3615-aa10-be56fe2c9ffa` |
| `eK6iyb` | `deadbeef-0d9e-3108-8fa9-4c02b120bbec` |
| `2r7unA` | `deadbeef-e474-35d1-9bdb-37965c294636` |
| `s0_rEa` | `deadbeef-4003-3292-9776-0061b1d33ab6` |
| `BLatD3` | `deadbeef-2a03-3968-8b67-08f726c6e47e` |
| `T4aJjK` | `deadbeef-a8a1-3a0c-a964-24d178cf7676` |
| `p8NyGH` | `deadbeef-f710-3f09-8c6f-c26b58c41441` |
| `YRlT9m` | `deadbeef-66f6-3712-8e2b-3e45ca4a99af` |
| `UA_ysE` | `deadbeef-7655-377e-9eb6-85914fd13128` |
| `c0RCNJ` | `deadbeef-6e52-3842-bcf9-c0ec80c9e9c2` |
| `SEiFbU` | `deadbeef-a08a-39ed-a27e-e827b973976f` |
| `QKEmSe` | `deadbeef-9548-318b-846d-8f33bfd1915d` |

#### deadc0de

| Name     | UUID                                   |
| -------- | -------------------------------------- |
| `lCE6Sw` | `deadc0de-2ccf-30cd-8d7d-1ec3a4272c27` |
| `D3Q4DY` | `deadc0de-ae61-3395-aa7a-72217c2985c4` |
| `axtAGo` | `deadc0de-9749-3901-9a3f-533ed839467d` |
| `V8FzTg` | `deadc0de-9dec-3afc-8877-3f778e856625` |
| `iU3JM7` | `deadc0de-3a8f-3f49-9f4e-57aa4a13f04d` |
| `wCeLwp` | `deadc0de-7077-34d5-93d3-e93815e2dd00` |
| `QKKJjs` | `deadc0de-cc3f-3a9f-83fb-c0048826eb24` |
| `zJKQR3` | `deadc0de-7512-3409-9685-be537db94fa6` |
| `R3AhJ_` | `deadc0de-d7e5-3948-8af4-db44cb312c80` |
| `Xgnkof` | `deadc0de-9bcb-3f2c-9f8e-da8941be048a` |
| `25vVy6` | `deadc0de-0301-35ea-b651-ee0435d1685d` |
| `ll1Riw` | `deadc0de-6eb9-3a80-93bf-b480480504ca` |
| `q7jYFA` | `deadc0de-bef3-3f2d-aa53-1f7ce3ada127` |

#### fee1dead

| Name     | UUID                                   |
| -------- | -------------------------------------- |
| `ama9y0` | `fee1dead-b1e1-3a42-a07e-72813297ad98` |
| `ZXnun9` | `fee1dead-f79d-3167-b2e5-a2c7d77628d6` |
| `pifBhj` | `fee1dead-7de7-3661-abad-49c2b93867ba` |
| `9mVnKh` | `fee1dead-ee99-31a3-902a-2317438791e4` |
| `IqRT43` | `fee1dead-59b0-3a70-ae60-a656f4e66466` |
| `bpxZQ1` | `fee1dead-890b-3726-8f22-8075eb19b55d` |
| `jWWYFp` | `fee1dead-6a4a-3439-9b9e-7713d4668585` |

#### defec8ed

| Name     | UUID                                   |
| -------- | -------------------------------------- |
| `elHadX` | `defec8ed-1414-33ff-8d74-ea56b6f6289e` |
| `LCOaiC` | `defec8ed-1c7b-38d7-bcf7-2638463e263c` |
| `kAVdHV` | `defec8ed-4441-3d44-8249-2867010f5a12` |
| `tnBqYQ` | `defec8ed-2876-3e16-b079-30354f7ad545` |
| `mgVBDK` | `defec8ed-860d-3033-8099-61d811ff33c2` |
| `WZ2BHf` | `defec8ed-a13a-36c2-9337-ed7c0a860227` |
| `H0wzxn` | `defec8ed-b71b-34a0-be5a-874a65a72055` |
| `aBUIvs` | `defec8ed-053e-3491-9045-c714d18c34b3` |
| `k0FIx2` | `defec8ed-0749-302f-b32c-1c5d66883304` |
| `U9xzEa` | `defec8ed-41f2-3432-9da7-b8b8a5ee4471` |
| `mXHQ_j` | `defec8ed-1cb5-3321-b559-29bb36279b27` |
| `YCXXte` | `defec8ed-2dac-3001-b07a-932b2de3804a` |
| `YL9ZqB` | `defec8ed-b2a6-3a58-8460-1bea84e7fadd` |

### 6-9 字符数值最小的前 10 个结果

从 6 字符长度开始已经包含大量结果，特此列出每个长度下数值最小的前 10 个结果，供参考。

这些结果通过 `find_most_difficult.py` 脚本生成。

#### 6 字符长度

| Name     | UUID                                   |
| -------- | -------------------------------------- |
| `EhxMb9` | `00000000-27ce-35d0-b1ca-15b8d6f93fbf` |
| `6jrfhE` | `00000000-3f6c-3385-828c-5804781a8528` |
| `bbCkvU` | `00000000-60ae-3785-b705-4d469c372801` |
| `wPMzRW` | `00000000-6ea1-3fad-859b-f3eb1b8938ef` |
| `CS908r` | `00000000-7a8c-35e3-9472-7c967f86342c` |
| `scRJUi` | `00000000-81b1-30f2-91c1-30e4ce59048b` |
| `ZnZDeI` | `00000000-8aaa-383a-b858-ed346d756725` |
| `i8Erv8` | `00000000-8f77-3ac9-884d-45d1b820a120` |
| `eZMaFP` | `00000000-a7ee-30f0-966b-6345ae2ed75e` |
| `Rgk6q8` | `00000000-ab3f-3623-8297-32a7b01fce22` |

#### 7 字符长度

| Name      | UUID                                   |
| --------- | -------------------------------------- |
| `tiHSNRY` | `00000000-0002-3ff2-bda6-bf24a74ffc4e` |
| `A0RW91b` | `00000000-0035-3bea-88b6-608e90005504` |
| `qtbny_c` | `00000000-0043-3753-9bdc-8c8c908cdec3` |
| `5klGIQ4` | `00000000-0176-3662-a4b9-37b5eacc0fe2` |
| `MC5FSKG` | `00000000-0186-3d46-a4a1-cdce1dfe1835` |
| `pHgYVOe` | `00000000-019d-3439-99eb-4602324c9785` |
| `HorE1ce` | `00000000-01d5-3977-82eb-8688a804d985` |
| `G7x1Az1` | `00000000-0224-3c1c-9f26-91835e25fefa` |
| `f3CpfyU` | `00000000-0268-391c-b6ae-8457622b7721` |
| `FHuombh` | `00000000-0336-3ec2-82da-4e3bc11669a4` |

#### 8 字符长度

| Name       | UUID                                   |
| ---------- | -------------------------------------- |
| `ilAQpWC2` | `00000000-0003-35b1-9df2-7537067aa4ba` |
| `gGky2Cl8` | `00000000-0003-3e30-8d39-be87ed4fe318` |
| `CoxaMpGo` | `00000000-0004-38a4-bd57-29df1a90148f` |
| `oyaAPbIB` | `00000000-0005-331e-8996-9959d3e979f9` |
| `BuFyn84P` | `00000000-0005-39a9-b65e-625304f8b4ec` |
| `DFJ8H9CW` | `00000000-0006-3538-bd03-685013a37bbd` |
| `BYDL8EWI` | `00000000-0007-38f2-b716-d51a34a31b8e` |
| `kabuQpoD` | `00000000-000b-320b-ba34-8fff723ee5a2` |
| `g2MWP53k` | `00000000-000c-3504-addc-3da1a1e75d61` |
| `x6Jx7Gxa` | `00000000-000c-3714-92a7-fa35aa41b01f` |

#### 9 字符长度（不完整）

| Name        | UUID                                   |
| ----------- | -------------------------------------- |
| `00pD0Yk1A` | `00000000-0000-3710-b5b1-7f404c93943b` |
| `08HicY3Uv` | `00000000-0001-3a4a-a0d2-2551b962bc8b` |
| `0aEw4t2_x` | `00000000-0005-35cc-9c82-cfa373df2a01` |
| `1KqjTy2Z9` | `00000000-000a-3376-9da4-8fd5cebeb527` |
| `1kZoqytSp` | `00000000-000b-39b3-ab01-91440c7f9c0a` |
| `0bHYkqtDE` | `00000000-000c-32ed-943d-11c4b923b4fb` |
| `02Pj0su51` | `00000000-0011-3aef-953b-79b4bf6808b4` |
| `0jIYOs7I0` | `00000000-0014-3e31-952c-fa3a095d7cb0` |
| `0APQIM3Kc` | `00000000-0016-36f8-91f7-9b2dc671b7c0` |
| `0oFwFMtgz` | `00000000-0016-3eb7-9600-1a2cd56fb5ed` |

## 许可证

MIT License
