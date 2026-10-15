# MLA 当前状态

当前分支 `feat/mla-cann-migration`。**M10c8a BSND PA 原生自动计划与 v24 安全协议已闭环。** 本轮基线为 `f65820ad05d8e1e986db22d125e0638886ce6022`；闭环提交用 `git log -1 -- docs/mla/status.md` 定位。唯一下一步为 [M10c8b：BSND PA 原生自动生产接入](plan.md#唯一下一步-m10c8bbsnd-pa-原生自动生产接入)，本轮不开始该步。

## 本步产物与开放边界

BSND Q + ND PA KV 的独立原生自动 CPU 计划、静态预算、v24 构造与 CPU/AICore 共用消费/查询协议已完成。保留固定 CANN PA 的 M/N 几何、每 M 变长 AIC 分区、FD 持久槽和 8 行 Vector 所有权；支持三种 mask、实际 KV/Q-used、空容量与原 Q allocation 地址。v24 为 `756+7*B` 字，独立校验当前长度、完整 cache 描述符和所有有效页，合法页表/stride 更换不改变记录身份。详见 [design 6.2.27](design.md#6227-bsnd-pa-原生自动计划与-v24-安全协议)。

本步复用既有原生任务/行地址查询，并从共享消费者提取独立分区校验，v18/v20/v22 的既有行为经重建后的累计 E2E 验证。静态预算覆盖计划 scratch、最多 `2*AIC` 个 FD 持久槽及独立 FP32 accumulator 上界；后续生产的每 AIC Direct 临时槽须另行预算。

公开能力保持上一闭环：BSND PA 单 split 与显式 `num_splits=2..256` 可用；三种非 PA 布局显式/自动及既有 mask 行为保留。**PA 自动 `num_splits=0` 与 TND PA split 继续明确拒绝。** v24 尚未编入 AICPU 或生产 launcher，不能将本步 CPU 证据表述为新路径数值、真实 Trap 或公开性能通过。未扩展 192/128、其他调度选项或完整图重放；CANN/CATLASS 未修改。

## 环境与证据身份

- Python 3.11.15、torch 2.7.1+cpu、torch_npu 2.7.1、CANN `/usr/local/Ascend/cann-9.0.0-beta.2`，6 号 Ascend910B3。CATLASS `769cd40a8716b28650b6bebb08db4834eea4462f`；固定 CANN `ca18aee610729ca5270a44e2d84c751bc687fbf0`，固定 FA3 与原有约定一致。
- 实际加载 `/home/zhangqijia/repo/flash-attention-npu/flash_attn_npu_3/flash_attn_npu_3.cpython-311-aarch64-linux-gnu.so`，SHA-256 `fe99163f6889800a810b20c9681ed3da973577aa48b461defb6a09e1c238c76b`。源码、编译依赖、链接产物和实际加载由本轮统一驱动记录并校验。
- 证据根 `/tmp/mla-m10c8a/`；构建和数学缓存沿用 `/tmp/flash-attention-npu-mla-46928a793876/mla-validation-artifacts/`。`build/` 只保留 `mla-codex-loop/` 日志、锁与恢复状态。证据位于 `/tmp`，缺失后不可继承。
- 接手时工作树干净、HEAD 与上步闭环一致，无遗留 MLA 验收进程，6 号卡空闲。`takeover.json` 复核上步累计 E2E 的 7,346 个源码/依赖/产物/原始证据文件，无差异；上步最小闭环命令和结果仍对应其基线。本步共享源码变化，旧 E2E 不继承。

## 本轮实际验收

全部使用 `python3 -m tests.mla`，先 dry-run/collect-only。下表最终有效组均为 0 failed、0 skipped；所有 CPU 单元均本轮实际执行，历史单元继承 0，历史 E2E 继承 0。

| 组 | 结果 | 耗时 | 证据目录（相对结果根） |
| --- | --- | --- | --- |
| auto-split-paged-bsnd-protocol-tools | 14 passed | 9.2 s | check-fixed/attempt-1792033125982726400 |
| auto-split-paged-bsnd-protocol-check | 13 passed | 43.1 s | 同上 |
| unit-auto_split_paged_bsnd_protocol | 407 passed | 44.7 s | cpu-final/attempt-1792033203337139090 |
| auto-split-paged-bsnd-protocol-sanitize | 407 passed | 145.6 s | sanitize-host/attempt-1792033401559699500 |
| 新扩展构建 / 实际加载 | passed | 533.5 / 21.2 s | e2e-final/attempt-1792033271188555370 |
| e2e | 215 passed | 41.7 s | 同上 |

- 固定源 PA tiling、AIC/FD/Vector、workspace 与 FA3 accumulator oracle 本轮重新编译执行；独立枚举三 mask 的可见行、逻辑 tile/page、FD 持久槽和原 allocation 地址。覆盖 P=16/112/512/528/1024、非整除 M、尾行/尾页/跨页、1/7/26 AIC、Direct/Partial/Empty、空值/Q-used、超过 uint32 的 group 与地址、精确 accumulator/总 workspace 溢出。协议覆盖双 dtype、损坏 header/端点/FD/Vector/宽前缀、陈旧长度/坏页、完整资源/描述符及查询失败原子性。
- 代表与成本检查先通过后固定实现；普通与 ASan/UBSan 是同一组 407 项，两组分别与自身收集顺序一致。最长 CPU 元数据输入的溢出检查包含双 dtype、零/非零逻辑容量；设备数值算法和数学执行策略未改，不重复计算历史最长数学 reference。
- 累计日常 E2E 保持 215 项，覆盖已接通路径。新模块尚未接入生产，其 E2E 在 M10c8b 接通时新增；本次不以旧无 mask/非 PA E2E 替代新协议的源差分/边界验收。数学 reference 共 **361 hit，0 重算**，待测实现全部实际执行。
- `selection-audit.json` 核对 247 个既有测试组/命令和日常选择均未改变；仅新增本模块工具、代表、所属单元和 sanitizer 入口。`cpu-evidence-audit.json` 核对 1,214 个源码/原始文件及收集/执行清单，无差异。`e2e-evidence-audit.json` 另核对 7,357 个 E2E 源码、构建/运行依赖及原始文件，无差异，215 项执行清单与收集一致。
- 本轮未运行历史组件单元、完整穷举、ordinary、v24 设备数值/Trap/公开性能、910C 实机或完整图重放，不计通过。普通 FA3 入口/调度/数值未改，本步相关设备回归为累计 E2E。

## 环境失败与收尾

开发时新组注册语法和工具的所属路径断言已修复，最终工具与代表组通过。沙箱内 E2E 收集因 NPU 不可见失败，宿主环境重做收集并完成实际执行；沙箱 sanitizer 的 407 项断言后，LeakSanitizer 在退出检查报环境错误，整组记失败。保持 ASan/UBSan/泄漏检查开启，在宿主环境完整重跑同一集合并通过。失败日志保留在 `dry-dev.log`、`check-dev/`、`collect-final/` 和 `cpu-final/`，均未转记为通过。

本次只核对 v24 记录/计划、共享消费者和相关验收契约；design 新增 6.2.27 与协议表，plan 推进唯一下一步。提交前检查记录见 `quality-final.log`、`doc-links-final.json` 和最终进程/设备记录；不为写自身 commit ID 再提交。

复现命令（各选择先加 `--dry-run`、再 `--collect-only`；sanitizer 与设备组需可用宿主环境，输出用新目录）：

```bash
python3 -m tests.mla --suite auto-split-paged-bsnd-protocol-tools auto-split-paged-bsnd-protocol-check --output /tmp/mla-m10c8a/repro-check
python3 -m tests.mla --unit auto_split_paged_bsnd_protocol --output /tmp/mla-m10c8a/repro-unit
python3 -m tests.mla --suite auto-split-paged-bsnd-protocol-sanitize --output /tmp/mla-m10c8a/repro-sanitize
python3 -m tests.mla --suite e2e --device 6 --output /tmp/mla-m10c8a/repro-e2e
```
