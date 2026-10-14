# MLA 当前状态

当前分支 `feat/mla-cann-migration`。功能基线仍为 M09d：512+64 的无 mask 五布局、causal 五布局，以及 BSND/TND/混合非 PA 与 BSND PA window 已接通；TND PA window 是下一步 M09e，split-KV 等后续能力仍未开放。本步只整理测试和工作产物，不修改生产算法、CANN 源仓或 CATLASS。

## 最近完成：单元测试 / E2E 整理

- MLA 用例按归属放到 `tests/mla/unit/<模块>/` 与 `tests/mla/e2e/<能力>/`，共享构建、固定源、独立 reference、性能与汇总驱动在 `support/`。`python3 -m tests.mla --list` 查看实际文件清单。
- 默认运行累计 E2E，多个模块合并、去重并共用最多 100 个节点；单元测试只在本模块开发或定位回归时显式运行。`--resume` 只继承内容及依赖完整的 E2E 正确性证据，单元和性能实际执行。旧命名组和 `tests.mla_validation` 保留兼容诊断入口。
- 原 25,602 个节点在纯目录迁移时逐一核对，无增删。随后将 causal BSND PA、causal TND PA、window BSND PA 的页大小内部循环改为 64 个显式参数，输入集合不变；日常 E2E 不自动选择最长 reference 成本检查，显式完整组仍保留。
- 原生性能所需 CANN 包有仓库内重建入口 `python3 -m tests.mla.support.native_package`，从固定 Git archive 构建，不再依赖旧步骤目录。只保留必要第三方下载缓存到外部工作目录，旧测试产物与结果不整体搬迁。

## 验证与产物

当前整理证据在 `/tmp/mla-test-reorg/`；生产/probe/reference 缓存与原生 CANN 包在 `/tmp/flash-attention-npu-mla-46928a793876/`。默认路径由 `MLA_WORK_DIR` 控制；需要长期继承时设为持久目录。`/tmp` 被清理后须重建并重测，不能把路径记录本身作为通过证据。

当前验证记录如下；没有把未选中或中断的组计为通过。

| 检查 | 结果 | 证据（相对 `/tmp/mla-test-reorg/`） |
| --- | --- | --- |
| 纯移动清单 | 原 25,602 节点一一对应，无增删 | `before-collect/`、`after-collect/` |
| 最终清单 | 25,993 节点；6 个页循环容器展开为 384 节点，另新增 9 个工具和 4 个来源检查；原页输入范围保留 | `final-collect-complete/`、`collection-comparison.json`、`final-collection-delta.json` |
| 限量工具组 | 100/100，73.6 秒；含成功/失败/中断传播及证据失效 | `tools-final/summary.json` |
| 新组织规则 | 9/9，4.1 秒；模块去重/分层、默认 E2E、种子、仅 E2E 恢复、外部产物、原生包身份与损坏检查 | `organization-final.log` |
| 原生包来源 | 10/10，2.7 秒；固定归档不依赖工作区 HEAD，错误头文件、缺少固定提交和未加载库仍拒绝 | `provenance-final.log` |
| CPU / ASan+UBSan 代表 | 各 6/6，34.4 / 113.8 秒 | `cpu/summary.json` |
| window PA E2E 代表 | 6/6，318.3 秒，FP16/BF16 源/双数学与公开/生产调用 | `representative/summary.json` |
| 独立错误子进程 | 2/2，65.6 秒 | `trap/summary.json` |
| 普通 FA3 | 100/100，40.7 秒 | `ordinary/summary.json` |
| 公开性能与原生分派 | 3/3，979.4 秒；双 dtype 共 24 配置，10 个适用配置平均比 0.377218，0 退化；14 个原生 FD 配置单列；两个 dtype 的组件采样通过 | `performance/summary.json`，本次 attempt 的 `window-paged-profile/public-profile/average.json` |
| 仓库质量 | 通过，含最终文档更新；`git diff --check` 通过 | `quality-final.log` |

完整生产扩展、PA probe 和 AICPU producer 已在外部目录重建并核验实际加载；后续设备组复用同一有效构建，构建确认约 1 秒。独立固定源 probe 的冷编译仍有成本；本次不改算法，不把全部历史模块的冷编译纳入必跑门槛。额外启动的全模块数值试跑已主动中断，保留在 `device/`，不计为通过；原定受影响路径由上表代表及公开性能验证覆盖。

性能前两次尝试分别遇到设备初始化超时和旧来源校验要求 CANN checkout HEAD 固定的问题，均未计为通过。当前校验使用固定 Git 对象与包内头文件，不改只读源仓；修复后的完整性能组通过。冷参考计算是主要耗时来源，尤其长 KV、128 头配置；已写入内容缓存。全量收集需正常环境查询设备型号，沙箱内失败单列保留，正常环境重收集通过。

`build/` 已仅剩 `mla-codex-loop/`：删除 133 个旧顶层项，释放 **197.54 GiB**；保留约 **243.3 MiB**，730 个日志/锁/状态文件 SHA-256 与清理前完全一致。清理前确认无旧目录在用、无未完无头轮次，清理后继续从新产物目录实测。详情 `cleanup.json`，旧产物没有整体转移到其他目录。

环境：Python 3.11.15，torch 2.7.1+cpu，torch_npu 2.7.1；CANN 9.0.0-beta.2；CATLASS `769cd40a8716b28650b6bebb08db4834eea4462f`；固定 CANN 源 `ca18aee610729ca5270a44e2d84c751bc687fbf0`。实机验证使用空闲 910B 设备 6，实际加载扩展及哈希由新运行的 `runtime.json` 记录。ASan/设备探测需要正常进程环境，沙箱 ptrace 中的 LeakSanitizer 失败已区分记录。

旧 M09d 及之前的详细结论可从 Git 中历史 status 查询。按用户要求清理 `build/` 后，旧原始证据不再可继承，也不把它们计入本步通过数量；生产能力的历史交付边界保持不变。

## 唯一下一步

本次整理完成，提交后立即停止。下次按 [plan](plan.md) 推进 M09e：TND Q + ND PA KV window 只读 cache。本模块新增自己的单元测试，接通后新增 E2E；后续回归只累计 E2E，失败再单点定位。不要恢复旧计划中的历史组件继承清单或逐层重复验收。
