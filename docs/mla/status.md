# MLA 当前状态

当前分支 `feat/mla-cann-migration`。功能基线仍为 M09d：512+64 的无 mask 五布局、causal 五布局，以及 BSND/TND/混合非 PA 与 BSND PA window 已接通；TND PA window 是下一步 M09e，split-KV 等后续能力仍未开放。本步按用户要求精简测试，不修改生产算法、CANN 源仓、CATLASS 或性能计时协议。

## 最近完成：按覆盖目的精简测试

- 取消正确性默认 100 节点预算。日常 E2E 只收集 `tests/mla/regression_cases.py` 的明确清单，选中即全跑；共享公开测试不加载历史组件 probe。单模块用 `--e2e <能力>`，多个能力合并去重。
- 当前日常 111 个节点：15 种已开放公开行为 × 2 dtype × 3 场景（长度、stride/复用、空 KV）为 90 个数值节点，另有 15 个 fake、4 个 compile、2 个未开放参数拒绝。使用原双数学 reference 与阈值；数量由独立覆盖目的决定，不是新的固定配额。一个能力通常为 7–8 个节点，dense 为 10 个。
- 模块专项在生成 pytest 节点前保留协议参数两两组合、独立 shape 的边界/角点、每个独立错误及 dtype；64 页 sweep 改用 8 个代表边界，公开专项去掉重复内部适配入口。保留 fixture、mark、nodeid 和原输入 seed。明确的高阶回归用具名 tuple 或 `mla_full_combinations` 保留。
- `--suite detail-<能力>` / `--unit <模块>` 显式运行专项；`--full-matrix` 恢复穷举。原历史定义保留，并未物理删除；默认不会带入日常回归。`--max-cases` 仅用于临时诊断，不是正式覆盖标准。仅有效 E2E 正确性结果可 `--resume`，精简/完整模式变化会失效。
- 无头提示和 AGENTS/design/plan 同步；新增行为须显式注册自己的日常 E2E。完整矩阵、长 reference、Trap、性能只按影响范围单列，不自动作为每一步门槛。

## 数量与本次验证

证据在 `/tmp/mla-test-slim/`，基线为上一闭环 `a648518`。完整清单仅做收集核对，没有把未执行用例记为通过。

| 清单 | 原节点数 | 当前默认展开数 |
| --- | --- | --- |
| 模块单元 | 9,720 | 6,564（含新增 5 个工具用例） |
| 公开专项正确性 | 16,250 | 6,301 |
| 显式 E2E 性能 | 23 | 23 |
| 新日常 E2E | — | 111 |
| 全仓 MLA 合计 | 25,993 | 12,999 |

默认日常只收集上表的 **111** 个。显式完整收集为 26,109 个，包含原 25,993 个全部节点、新日常 111 个及新工具 5 个；没有丢失旧节点。清单与差异在 `compact-final/`、`full-final/`、`inventory-comparison.json`，各模式收集约 34 秒。

| 检查 | 结果 | 证据（相对 `/tmp/mla-test-slim/`） |
| --- | --- | --- |
| 生成规则、归属及选择/性能策略 | 43/43，57.7 秒 | `tools.log` |
| 最终生成规则及驱动成功/失败/中断/证据失效 | 31/31，48.8 秒；与上行存在重复，不相加 | `runner-tools.log` |
| 新增完整模式身份失效及环境隔离检查 | 2/2，3.7 秒 | `full-mode-tools.log` |
| 日常 E2E 全集 | **111/111，无 skip/失败；执行 28.6 秒，构建校验 1.2 秒，设备/加载检查 28.0 秒，合计约 58 秒** | `daily/summary.json`；有效 attempt 为 `attempt-1791966630484100490` |
| 页边界专项 | 32/32，28.5 秒；8 页 × 双 dtype × 公开真实/fake，复用相同已核验扩展 | `page-representatives/summary.json`、`command.json` |
| 无头日志保留 | 730 文件 SHA-256 与整理前完全相同，build 仅有 mla-codex-loop | `headless-check.json` |
| 仓库质量 | 暂存文件质量检查、文档链接及 git diff --check 通过 | `quality.log` |

初次日常尝试发现新测试对未开放参数错误消息匹配过窄，已按既有拒绝契约修正；随后一次准备阶段因格式化修改测试源码被驱动判为失效。两次均未计为通过，日志保留在 daily 的早期 attempt 中；实现固定后上述完整 111 个通过。没有放宽精度阈值或屏蔽失败。

生产/probe/reference 缓存在 `/tmp/flash-attention-npu-mla-46928a793876/`，核对内容后复用。上一轮公开性能记录在 `/tmp/mla-test-reorg/performance/`（24 配置，10 个适用配置均值 0.377218、0 退化，14 个原生 FD 单列）；本步未改输入/计时协议，仅保留历史结果，不计本次通过或继承。没有重新运行完整数值矩阵、CPU sanitizer 或性能矩阵。

环境：Python 3.11.15，torch 2.7.1+cpu，torch_npu 2.7.1；CANN 9.0.0-beta.2；CATLASS `769cd40a8716b28650b6bebb08db4834eea4462f`；固定 CANN 源 `ca18aee610729ca5270a44e2d84c751bc687fbf0`。实机为 910B 设备 6；实际扩展路径、哈希和健康检查见有效 attempt 的 `runtime.json`。`MLA_WORK_DIR` 可设持久目录，临时证据丢失后不得继续继承。

复现日常命令（单模块选 `--e2e window_paged` 时为 7 个）：

```bash
python3 -m tests.mla --dry-run
python3 -m tests.mla --collect-only --output /tmp/mla-step/collect
python3 -m tests.mla --device 6 --output /tmp/mla-step/e2e
python3 -m tests.mla --e2e window_paged --device 6 --output /tmp/mla-step/affected
```

## 唯一下一步

本次精简提交后立即停止。下次按 [plan](plan.md) 推进 M09e：TND Q + ND PA KV window 只读 cache。本模块新增必要单元测试，接通后注册小输入日常 E2E；其他模块只走受影响 E2E，不恢复历史组件累计验收或固定数量配额。
