# MLA 当前交接

## 状态与唯一下一步

- 当前步骤：M04b2b2b2b2b2a **已完成探测与回归验收，未合入 kernel 优化**。最终产物是 36 项跨 64 列半区的极值回归及两种 softmax 候选的实测结论；本次闭环提交可由 `git log -1 -- docs/mla/status.md` 定位。
- 当前分支：`feat/mla-cann-migration`。生产代码与 `23d7304` 一致，支持连续 BSND、512+64、FP16/BF16、KV `1..1280`、显式 `num_splits=1`。`1281`、`8193` 等更长 KV 继续明确拒绝。
- 完整 tile 的整段 softmax 特化，以及仅直接合并两半最大值的两个候选均未证实收益，已撤回。不能把本步记为优化上线或长 KV 已支持。
- 唯一下一步：M04b2b2b2b2b2b，优先定位 Cube 与长 KV 流水，再优化并验收 `KV >1280`，含 8193 的单/多 query、FP16/BF16 O/LSE、重复执行及同轮 CANN 性能；保持显式 `num_splits=1`。本次提交后停止。

## 最终合入内容的闭环证据

| 检查 | 结果 |
| --- | --- |
| 接手 | PASS：最初工作区干净，复核上一步 457 项证据 checksum、34 项源码/构建/测试哈希及两个源仓状态 |
| 基线恢复 | PASS：生产源码恢复；重新构建的全部 14 个对象与 `23d7304` 逐字节一致，未保留候选 kernel。扩展重新链接，对象顺序与此前不同，实际 SHA 记录如下 |
| 新增回归 | PASS：恢复基线上 36 项，覆盖两个 64 列半区首尾极值、正负 query、跨 tile/batch 最大值变化、1153/1279/1280、1/3/8/16/17/33 头及三次重复执行；原双 reference 阈值未改 |
| 普通 FA3 | PASS：同一恢复扩展的 11 项精度、metadata、编译图回归；普通回归来自恢复后的 `baseline-regression.xml`；极值构造加强后只重跑受影响的 36 项，最终合计 47 个不同节点，无 skip |
| 构建及测试完整性 | PASS：实际加载扩展与恢复构建哈希匹配，旧测试 AST 未改；CATLASS 与 CANN 源仓状态保持接手时原样 |
| 最终设备记录 | PASS：本轮授权的物理 2 号共享 Warning 卡；两次最终基线运行分别保存 57 / 49 个快照，前后告警均为 81078603、ECC 八项计数均为 0。不得视为健康或独占设备 |
| 提交质量 | PASS：`git diff --check` 与 `ci/code_quality.sh --staged` 的 5 项适用检查；仅暂存本步四个测试/文档文件，保留仓库提交钩子 |

当前提交只改变测试和文档。候选运行的更大测试集记录在下一节，不能混作恢复后又重跑了全量 MLA 或上线了新的性能优化。

## 两种已撤回候选的结论

| 候选 | 数值和性能证据 | 结论 |
| --- | --- | --- |
| A：完整 128 列 tile 的整段 softmax 特化，直接合并两半最大值 | 736 个不同 MLA 节点、186 组固定 CANN O/LSE 与分派、11 项普通 FA3 通过；100 组旧新 O/LSE 逐位一致；56 例同轮 CANN 比较逐例中位数与 95% 区间上界小于 1，56 次分派均为固定源码 MLA template 5 | 有效的 64 例旧新 kernel 比较为 0 改善、2 微退化、62 区间跨 1。`rows1-float16` 和 `rows7-float16` 比值分别为 1.01423、1.00840；不采用 |
| B：仅在完整 tile 直接合并两半最大值，保留通用 softmax 主体 | 12 组 O/LSE 逐位一致；重点覆盖前述退化项、D01、1280 单/多 query，两种 dtype。每例每实现 150 个 kernel 样本，12 例 95% 区间均包含 1 | 未证实收益，不继续扩大验收，已撤回 |

A 的首轮 100 例旧新采集在计时窗口出现四个外部进程，性能无效，数值证据仍有效；上述 64 例为另行有效重测。A 的 56 例 CANN 比较及 B 的 12 例探测均通过各自设备窗口检查。没有放宽精度或性能阈值，也没有将区间跨 1 记为提速。

最初一次针对性精度启动被旧占用检查拦截，没有执行 pytest；重新记录本轮实际共享占用后完成测试。上述中断、无效采集和未采用候选均保留原始记录。

## 运行环境与设备授权

用户本轮明确授权“允许本轮使用 2 号卡，记录告警和占用”，例外见 design 第 7 节。使用物理 2 号、进程内 `npu:0`；设备仍为 Warning，错误码 `81078603`，内容为端口链路 `up to down`，没有恢复健康的结论。

最终基线测试开始时 AICore 0%，已有后台 PID `3724055` 占 140 MB；所有快照、运行 PID 及前后健康/ECC 输出保存在 monitor 文件。早期功能与无效性能运行中的额外占用另有完整记录。本轮授权不扩展至下一步。

| 项目 | 实际核实 |
| --- | --- |
| Python / 包 | `/usr/local/python3.11.15/bin/python` 3.11.15；torch `2.7.1+cpu`，torch_npu `2.7.1` |
| 芯片 / CANN | Ascend910B3；`/usr/local/Ascend/cann-9.0.0-beta.2` |
| 固定 FIA / CATLASS | FIA `ca18aee610729ca5270a44e2d84c751bc687fbf0`；CATLASS `769cd40a8716b28650b6bebb08db4834eea4462f`，子模块干净 |
| 实际加载扩展 | 本仓 `flash_attn_npu_3/flash_attn_npu_3.cpython-311-aarch64-linux-gnu.so` |
| 最终恢复扩展 SHA256 | `e267a7fb331430dd1e1aeca5becd6ee3c6b5cd07f7c4c0756d76c4579306006b` |
| 原扩展备份 SHA256 | `81f61d518d8e2ced93d6f4a22ee31a5f3166598e7b24b3b4e0b7e3c61c6b94b4`；两个扩展的 14 个构建对象相同，重链顺序证据已保存 |

## 持久证据与复现

证据根目录：`/home/zhangqijia/repo/flash-attention-npu/build/mla-m04b2b2b6-validation/`。

- **最终合入验收**：`restored-baseline/` 的 `build-restored-*`、`build-provenance-restored.json`、`link-order-proof.json`、`environment-restored.json`、`baseline-regression*`（含 11 项普通 FA3）、`baseline-maxima-final*`（最终 36 项）、`test-source-initial.json` / `test-source-final.json`、`acceptance-integrity.json`、质量日志。
- **候选 A（未采用）**：根目录的 `mla-coverage.json`、`cann-accuracy.json`、`dispatch-accuracy-proof.json`、`compare-versions-final*`（首轮性能无效）、`compare-versions-retry*`、`versions-summary.json`、`compare-cann-final*`、`performance-acceptance.json`、`dispatch-performance-proof.json`、`acceptance-integrity.json`（仅 A 的验证）；`full-tile-candidate/` 保留源码、patch、扩展与撤回决定。
- **候选 B（未采用）**：`direct-max/` 的源码、patch、扩展、构建溯源、`compare-probe*`、`versions-summary.json` 和撤回决定；没有记作完成全量候选验收。
- `takeover-audit.json`、`test-source-hashes.json`、`probe-test-source.py`（候选与首轮基线使用的原始极值构造）、`functional-device-summary.json` 及最终 `checksums.sha256` 保存来源和完整性。固定源码 CANN 包仍为 `build/mla-m02b-validation/opp-dense/`；此前已闭环的 700 项及 56 例性能证据见 `23d7304` 的 status。

最终最小实机命令如下；普通 FA3 的九个节点展开为 11 项，精确节点保存在 `run_baseline_regression.py`。复跑使用新标签和输出目录，不覆盖原证据；重新检查设备、后台占用及扩展哈希，使用 `login=false`、沙箱外并保留 CANN PYTHONPATH。

```bash
python build/mla-m04b2b2b6-validation/restored-baseline/monitored_run.py baseline-maxima-final \
  timeout -k 15 600 python -m pytest -q -x \
  tests/test_flash_attn_npu_v3_mla_online.py::test_online_maximum_across_halves \
  --junitxml=build/mla-m04b2b2b6-validation/restored-baseline/baseline-maxima-final.xml

python build/mla-m04b2b2b6-validation/restored-baseline/verify_restored.py --staged
```

候选的实际构建、精度及性能参数均保存在对应 `*-command.json`，不作为新功能接通命令。质量工具仍使用 `/tmp/mla-m01-precommit`（PYTHONPATH）和 `/tmp/mla-m01-precommit-cache`（PRE_COMMIT_HOME）。
