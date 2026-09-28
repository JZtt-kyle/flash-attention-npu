# MLA 当前交接

## 状态与唯一下一步

- 当前步骤：M04b2b2b2b2b1 **已完成并闭环**，连续 BSND、512+64、FP16/BF16 MLA 支持范围扩至 KV `1..1280`，保持显式 `num_splits=1`。本次闭环提交由 `git log -1 -- docs/mla/status.md` 定位。
- 当前分支：`feat/mla-cann-migration`。本步沿用 `1378760` 的 128 列在线 softmax 和三槽 KV 预取，仅放宽 Python/fake/C++ 上限并补齐边界测试；kernel 实现、workspace、metadata、CATLASS 和 CANN 源仓未修改。
- 尚未支持的组合继续明确拒绝，包括 `1281`、`8193` 等 `KV >1280`；本步不代表整个 FIA/FA3 已迁移。
- 唯一下一步：M04b2b2b2b2b2，优化并验收 `KV >1280` 的长 KV 调度，包含 8193 的单/多 query、FP16/BF16 O/LSE 与同轮 CANN 性能；保持显式 `num_splits=1`。本次提交后停止，下一步尚未开始。

## 最近闭环证据

| 检查 | 结果 |
| --- | --- |
| 接手与构建一致性 | PASS：本次复核此前 340 项证据 checksum、34 项源码/构建/测试哈希及两个源仓状态；实际加载扩展与功能验证相同。最终构建中 13 个 kernel/AICPU 对象与 `1378760` 逐字节一致，仅 `flash_api.o` 因上限变更而变化 |
| MLA 精度与契约 | PASS：700 个不同节点，online 326 + short 141 + contract 233，无 skip；原双 reference 阈值未改 |
| 固定源码 CANN 精度/分派 | PASS：150 组新实现/CANN 的 O/LSE 双 reference 检查，150 次 FIA launch 匹配固定源码配套二进制 |
| 普通 FA3 | PASS：11 项精度、metadata 和编译图回归，无 skip；pytest 31.01 秒 |
| 同轮性能 | PASS：56 例 kernel 和普通调用逐例不劣于 CANN，逐例配对 bootstrap 95% 区间上界均小于 1；本轮同时重查 56 例 O/LSE |
| 性能输入的分派来源 | PASS：另用同一 56 个输入记录 56 次 FIA launch，全部为固定源码的 MLA template 5；包及候选扩展与计时运行一致 |
| 设备记录 | PASS：用户授权的 2 号共享 Warning 卡；性能运行 52 个快照，其中覆盖计时窗口及边界的 8 个快照未出现额外进程；前后 ECC 八项计数全为 0 |
| 提交质量 | PASS：`git diff --check` 与 `ci/code_quality.sh --staged`；只暂存本步 8 个文件，保留仓库提交钩子 |

700 节点通过来自两轮：首轮 `641 passed, 1 failed` 后停止，其中 online/short 的 467 项全部通过。失败原因是公共入口拒绝测试仍使用旧超界长度 `17 × 61 = 1037`，该长度已在新范围内；将它改为 `17 × 76 = 1292` 后重跑全部 233 项契约通过。失败 XML、修改前测试和去重覆盖证明均保留，不记为单轮全过。本次性能运行前后产品源码、扩展及测试哈希均未再改变；未重复执行未受影响的 700/150/11 项检查。

以下均为 `新实现 / CANN` 的耗时中位数比值：

| 范围 | 用例数 | kernel 比值 | 普通调用比值 |
| --- | --- | --- | --- |
| KV 1..1024 代表用例 | 32 | `0.537..0.814` | `0.833..0.891` |
| KV 1025..1280 代表用例 | 24 | `0.710..0.949` | `0.795..0.935` |
| 全部 | 56 | `0.537..0.949` | `0.795..0.935` |

最接近门槛的 kernel 用例为 `kv1279-q1-bfloat16`：20.800 / 21.910 us，比值 0.94934。全部 kernel / 调用的 95% 区间上界最大值分别为 0.95930 / 0.95710，均可分辨不劣化，无需补测。每实现预热 10 次，调用计时 7 轮 × 50 对，profiler 3 轮 × 10 对，共 3360 个 kernel；同一输入交替 AB/BA，每次调用后同步。MLA 不使用 metadata，CANN 保持原生自动 split；没有调整精度或性能阈值。

此前 1025..8193 探测精度通过，但 1536 单 query 与 8193 存在性能退化，故本步收敛于 1280。曾尝试的完整 L0B 候选已撤回，其计时因新增后台进程无效；这些旧探测不能视为更长 KV 已支持。

## 运行环境与设备授权

用户先要求选卡测试，随后明确要求“就用当前的卡测一下性能，验收一下”。本步按 design 第 7 节记录的单步例外，在物理 2 号（进程内 `npu:0`）完成性能验收：健康状态仍为 **Warning**，错误码 **81078603**，内容为端口链路 `up to down`；此结果不表示设备已经恢复健康。

性能计时窗口为 `2026-09-28 08:03:49.446467..08:04:01.303141 UTC`。开始时 AICore 0%，已有后台 PID `3724055` 占 140 MB，本轮 driver PID 为 `1561693`；窗口及相邻边界快照仅见这两个进程。监控另保留计时前准备和计时后导出的快照，前后告警一致且 ECC 为 0。

先前功能运行曾在 `07:34:17` 观察到额外 PID `1149729`（169 MB），已记录于功能证据；该进程未出现在本次性能窗口。当前授权限定于本步，后续步骤仍按默认健康、空闲设备口径及新的明确授权执行。

| 项目 | 实际核实 |
| --- | --- |
| Python / 包 | `/usr/local/python3.11.15/bin/python` 3.11.15；torch `2.7.1+cpu`，torch_npu `2.7.1` |
| 芯片 / CANN | Ascend910B3；`/usr/local/Ascend/cann-9.0.0-beta.2` |
| 固定 FIA 源 | `ca18aee610729ca5270a44e2d84c751bc687fbf0`，已有源仓状态保留 |
| CATLASS | `769cd40a8716b28650b6bebb08db4834eea4462f`，干净且未修改 |
| 实际扩展路径 | 本仓 `flash_attn_npu_3/flash_attn_npu_3.cpython-311-aarch64-linux-gnu.so` |
| 扩展 SHA256 | `81f61d518d8e2ced93d6f4a22ee31a5f3166598e7b24b3b4e0b7e3c61c6b94b4` |

## 持久证据与复现

证据根目录：`/home/zhangqijia/repo/flash-attention-npu/build/mla-m04b2b2b5-validation/`。

- `card2-warning-perf-20260928T080123Z/` 为最新性能与闭环证据：`takeover-audit.json`、`environment-current.*`、`compare-card2-a.json` / `.log` / `-command.json` / `-exit.json`、`profile-card2-a/`、设备 monitor/proof、`performance-acceptance-a.json`、`dispatch-performance.*`、`dispatch-proof.json`、`acceptance-integrity.json`、质量日志及 `checksums.sha256`。
- `card2-warning-test-20260928T072906Z/` 为功能证据：`mla-final.xml`（首轮失败）、`contract-before-fix.py`、`mla-contract-fixed.xml`、`mla-coverage.json`、`cann-accuracy.json`、`dispatch-proof.json`、`regression-final.xml`、各阶段命令/退出码/设备记录和修正后的 `test-source-hashes.json`。其 handoff 文件保留当时“功能通过、性能待验收”的历史状态。
- 父目录的 `build-final-*`、`build-provenance-final.json` 保存当前构建和对象哈希；`long-probe*`、`wide-probe*` 保留范围收敛的原始探测。父目录及功能子目录旧 checksum 均保留，本次未覆盖。
- 固定源码包为 `build/mla-m02b-validation/opp-dense/`；原版 620 项接手复核及 `1378760` 的此前闭环证据仍在原位置。本步未改 CPU reference/profile/provenance 单测实现，未将未重跑的旧 CPU 检查计作本轮通过。

本轮实际性能命令如下；复跑使用新标签/目录，重新核对设备、后台占用、环境及扩展哈希，按实际条件调整运行器和报告字段。实机使用 `login=false`、沙箱外，并保留 CANN PYTHONPATH。

```bash
python build/mla-m04b2b2b5-validation/card2-warning-perf-20260928T080123Z/monitored_run.py compare-card2-a \
  timeout -k 15 600 python build/mla-m04b2b2b5-validation/card2-warning-perf-20260928T080123Z/compare.py \
  --output build/mla-m04b2b2b5-validation/card2-warning-perf-20260928T080123Z/compare-card2-a.json \
  --profile-dir build/mla-m04b2b2b5-validation/card2-warning-perf-20260928T080123Z/profile-card2-a \
  --iterations 50 --rounds 7

python build/mla-m04b2b2b5-validation/card2-warning-perf-20260928T080123Z/verify_monitor.py compare-card2-a

python build/mla-m04b2b2b5-validation/card2-warning-perf-20260928T080123Z/accept-performance.py \
  --input build/mla-m04b2b2b5-validation/card2-warning-perf-20260928T080123Z/compare-card2-a.json \
  --output build/mla-m04b2b2b5-validation/card2-warning-perf-20260928T080123Z/performance-acceptance-a.json
```

分派核对的完整命令见 `dispatch-performance-command.json`，最终综合检查为 `acceptance_integrity.py`。质量工具使用 `/tmp/mla-m01-precommit`（PYTHONPATH）和 `/tmp/mla-m01-precommit-cache`（PRE_COMMIT_HOME）。
