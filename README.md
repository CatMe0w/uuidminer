# uuidminer

中文版请见[这里](https://github.com/CatMe0w/uuidminer/blob/master/README_zh.md)。

uuidminer is a high-performance CUDA-based search tool designed to find offline-mode Minecraft player names that correspond to UUIDs with specific prefixes.

In offline mode, a player's UUID is determined by the MD5 hash of the string `OfflinePlayer:[playername]`, with the version and variant bits set according to UUID v3 rules. Since the player name space can be viewed as an enumerable search space, this project performs a parallel search on player names to filter out results where the UUID high bits match a specific pattern.

## Motivation & Evolution

This project started as a joke: if we treat player UUIDs like Bitcoin block hashes, could we define a "name difficulty" and find the "most difficult" player name?

Later, the idea became more practical: the program's performance improved to surpass hashcat. At the same time, the search target was no longer limited to the number of leading zeros, but allowed users to specify any target UUID prefix, such as the hexadecimal string `deadbeef`, or a combination of dates and numbers with personal significance.

Now, uuidminer is a general-purpose offline player UUID searcher, no longer limited to finding leading zeros.

## Features

The program enumerates the legal Minecraft player name character set (excluding CJK player names), i.e., 0-9, a-z, A-Z, and \_, totaling 63 characters. It searches for player names with lengths from 3 to 16. Each candidate name is mapped to its corresponding offline mode UUID and matched against the user-specified target prefix.

The program supports multi-GPU and multi-node parallel execution.

When no arguments are passed, the program outputs results if the UUID has at least 8 leading zeros. Users can also specify any 8-digit hexadecimal prefix as the search target.

## Performance & Status

In the current implementation, the measured performance of a single RTX 3090 is approximately 44 GH/s, which is about **70%** of hashcat mode 0 "MD5" and **126%** of hashcat mode 20 "md5($salt.$pass)".

The current implementation has been optimized close to the theoretical limit for this specific target. Due to the username generation logic, it cannot reach the peak performance of hashcat mode 0 pure MD5.

At this performance level, the search for all 8-character results has been fully completed. For 9-character length, a single 3090 would take about 4 days of continuous running, and there are currently no plans to exhaustively search it.

## Design Trade-offs on Leading Zero Threshold

The program uses 8 leading zeros as the output threshold instead of a longer 12, i.e., `00000000-xxxx-xxxx-xxxx-xxxxxxxxxxxx` instead of `00000000-0000-xxxx-xxxx-xxxxxxxxxxxx`, for the following reasons:

In actual searching, results with 12 leading zeros do not appear until entering the 9-character name space. In other words, purely for the visual effect of "two segments of leading zeros," the computational cost rises sharply while the return is extremely limited.

At the same time, due to the existence of UUID version and variant bits, the 13th character is fixed as `3`. From this point on, the visual "rarity" drops significantly. I personally believe that pursuing more leading zeros beyond 12 bits has no obvious display or research value.

The final consideration is that the results for the first 8 bits are abundant enough. Most target prefixes can find dozens of results within the 6-character or even 5-character space, which should be sufficient for most users.

## Analysis Tools

The project provides an additional Python script `find_most_difficult.py` for statistical analysis of the search results.

The script only analyzes the classic "leading zero" target, but you can modify it to analyze other prefixes or patterns.

## Distributed & Multi-node Configuration

The program supports running in multi-GPU and multi-node environments. All distributed behaviors are controlled via command-line arguments and do not rely on external scheduling mechanisms.

Core parameters are as follows:

-   `--node-count`: The total number of logical nodes in the entire cluster
-   `--node-index`: The node ID of the current instance, starting from 0
-   `--node-slices`: The number of additional continuous work slices claimed by the current node

For heterogeneous computing clusters, `--node-slices` is _super effective_. In an environment with unbalanced computing power, high-performance nodes can actively take on more search space by increasing slices without re-partitioning the entire cluster.

### Example 1: Homogeneous Cluster

Suppose there are 4 machines with similar performance, running:

```
--node-index 0 --node-count 4 --node-slices 1
--node-index 1 --node-count 4 --node-slices 1
--node-index 2 --node-count 4 --node-slices 1
--node-index 3 --node-count 4 --node-slices 1
```

Each machine is responsible for one-quarter of the search space.

### Example 2: Heterogeneous Cluster

Suppose there is 1 high-performance machine and 2 low-performance machines. To let the high-performance node take on half of the workload:

High-performance node:

```
--node-index 0 --node-count 4 --node-slices 2
```

Two low-performance nodes:

```
--node-index 2 --node-count 4 --node-slices 1
--node-index 3 --node-count 4 --node-slices 1
```

### Result Output & CSV Piping

The program outputs hits directly to standard output in the format:

`playerName,uuid`

To save the results as a CSV file:

`uuidminer.exe --target deadbeef > results.csv`

## Known Most Difficult Usernames

Below are some of the "most difficult" player name results mined so far.

Difficulty is defined as the UUID having the smallest numerical value.

| Length | Name        | UUID                                   |
| ------ | ----------- | -------------------------------------- |
| 3      | `ign`       | `00006dae-4a2b-3a71-8473-11a2d1553f3f` |
| 4      | `v4K4`      | `00000109-807f-3287-9a54-90aa57e7362a` |
| 5      | `16j63`     | `00000003-a5f8-3217-96d2-c96c3f60529c` |
| 6      | `EhxMb9`    | `00000000-27ce-35d0-b1ca-15b8d6f93fbf` |
| 7      | `tiHSNRY`   | `00000000-0002-3ff2-bda6-bf24a74ffc4e` |
| 8      | `ilAQpWC2`  | `00000000-0003-35b1-9df2-7537067aa4ba` |
| 9      | `00pD0Yk1A` | `00000000-0000-3710-b5b1-7f404c93943b` |

Note: The results for 9-character length have not been fully mined; the table only lists the currently known smallest result.

### Special Results

Results for some well-known [magic numbers](<https://en.wikipedia.org/wiki/Magic_number_(programming)#Debug_value>).

#### deadbeef

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

### Top 10 Smallest Numerical Results for 6-9 Characters

Since the 6-character length already contains a large number of results, the top 10 results with the smallest numerical values for each length are listed here for reference.

These results were generated by the `find_most_difficult.py` script.

#### 6 Characters

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

#### 7 Characters

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

#### 8 Characters

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

#### 9 Characters (Incomplete)

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

## License

MIT License
