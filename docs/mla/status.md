# MLA 当前状态

**M16 按用户明确授权接受本次图性能例外并提交验收。正确性、eager 和调度专项性能通过；图公开性能平均比 1.455079，仍记为未达标。** 分支 `feat/mla-cann-migration`，本步基线 `26145fa`（M15g2）。2026-10-18 用户要求“图模式的性能先这样吧。先提交验收一次”，本次保留性能失败及全部证据，不改变测试门槛。数值算法及 CANN/CATLASS 源码未改。本次作为最近专门文档轮后的第五个已接受开发轮，唯一下一步为 [专门文档修正轮](plan.md#唯一下一步-专门文档修正轮)；提交后立即停止，不在本次启动该轮。

## 当前实现

v47 自动任务过去用 partial 容量作调度步长，容量等于发射核数时，所有未拆分 head group 都落在核 0。现改用实际最大分片数枚举；原 partial 地址容量步长、源片段端点和显式 split 规则保持不变。新增 CPU 的任务集合/地址/核分布边界，及 BSND/TND 慢形状的固定源、双数学与日常 E2E。

初始化、partial attention、FD merge 合为一次发射；各核先完整验证 metadata，再执行初始化和 attention，所有 Vector 核按源 FlashDecode 的 SyncAll 顺序进入归并。短 Q 不再枚举固定 M tile 中不可能有行的归并任务。设备 loader 复用已校验的 layout，并展开固定 32-word 头比较，消除重复准备。数值 Kernel/Block/Tile、公开返回与 accumulator、当次长度/页表校验及 Trap 保留。

图捕获时旧 metadata producer 使用当前流，eager 保留原辅助流；profiler 在独立进程内先完成 eager/AICPU 初始化、首次捕获放在 RECORD，保留原始 Model ID 与完整任务校验。wall 单独关闭 profiler 计时；R、预热、重复次数、精度和平均门槛不变。范围仍仅 MLA，两种表示中的分组调度命名不表示迁移普通 GQA。

## 本轮证据与未通过项

本轮证据根 `/tmp/mla-m16-schedule/`，生产源码冻结后通过格式和质量检查（`quality-frozen.log`）。正式重建后，扩展哈希与构建 manifest、实际加载一致。计时生命周期修正后再次通过质量检查（`quality-wall-lifetime.log`），`wall-dry-run.log` 与 `wall-collect/summary.json` 已核对命令和集合。最新正确性验收位于 `final/attempt-1792339158554700900`，相关组全部通过；统一身份未允许继承旧结果，全部实际重跑，继承 E2E 为 0。磁盘问题处理后，仅重跑受影响的图性能组，最终证据为 `private-profile/attempt-1792342031198773140`，完整采样成功，平均图性能门槛失败。

本次收尾只同步用户授权的验收边界，生产、测试及构建代码未变。使用统一驱动已有校验核对 16 个通过组的源码/依赖、用例清单、实际加载与原始证据，并核对最终图组的失败结果及扩展身份（`commit-evidence-check.log`）；未重跑实机，不将核对记作新通过。对本步 30 个暂存文件执行 `ci/code_quality.sh --staged`，适用质量钩子全部通过（`quality-commit-staged.log`）；`git diff --check`、文档链接及更新后的下一步锚点核对通过。图性能按本次明确例外接受，失败结果不改写为通过。

| 证据 | 结果及适用范围 |
| --- | --- |
| `final-tools/attempt-1792334553417760170` | 当前生产实现的 CPU 30、sanitizer 119、调度工具 22 项全部通过；36.5/119.3/14.4 s；图工具由下行重跑替代 |
| `wall-tools/attempt-1792338966969631670` | 图工具 170 项通过，110.4 s，含新增结果析构和失败传播回归 |
| `collect/attempt-1792334533971256540`、`wall-collect/attempt-1792338890655943560` | 前后集合已核对；仅图工具新增 3 项，累计 E2E 仍为 871 项，性能仍为 40+56 配置 |
| 最新 final 的 build/runtime | 正式构建复核复用 1.2 s，实际加载核验 24.4 s |
| 最新 final 的 GQA 代表/固定源/接口边界 | 90/32/100 项通过，57.2/42.5/22.1 s |
| 最新 final 的图代表/图性能预检 | 13/1 项通过，28.7/305.7 s；预检不等于平均性能通过 |
| 最新 final 的调度专项/图专项 | 402/233 项通过，474.9/354.9 s，含生命周期和 Trap |
| 最终 private-profile 的 compressed FP16 图 Kernel | BSND auto window：整段融合 Kernel 中位数 77.492 μs，native 27.141 μs；两者均保留原始任务记录 |
| 最新 final 的累计 E2E/普通 FA3/普通图回归 | 871/504/2 项通过，218.7/30.2/23.5 s；旧 E2E 继承为 0 |
| `final/attempt-1792335232667283470` 的调度性能 | 56 配置已测，适用 20 配置平均比 0.726879，通过，690.9 s；配置退化 0、分段退化 13，28 个 single/explicit 原生 FD 和 8 个正 softcap 按既有口径单列；其驱动和生产实现此后未变 |
| 同一旧 final 的图性能 | 40 配置全部采样；旧驱动 eager 均值 0.784606、图均值 1.552423（失败）。前一 eager 结果释放污染图计时，wall 不作最终验收，保留原始 Kernel trace 用于定位 |
| 最新 final 的图性能 | 290.5 s 后在首个图 Kernel 采样器失败；磁盘空间不足导致 CANN 原始记录缺失，未产生修正后均值，不记为通过 |
| `wall-profile-initial/disk-probe` | 切换到临时内存目录后的真实预检通过：300 个图 Kernel 样本、1 份原始 op_summary，Model ID 与阶段严格校验通过 |
| `wall-profile-initial/attempt-1792341404969637850` | 完整采样 48.3 s 后中止：实机容器的 `/dev/shm` 仅 64 MB，native 预检日志填满目录；未产生性能结果，原证据已校验归档 |
| `private-storage-probe/sample/profiler-probe` | 独立 4 GB tmpfs 的真实 profiler 预检通过，97 个文件校验归档，原路径保持可访问 |
| `private-profile/summary.json` | 构建/实际加载复核 1.3/22.3 s；40 配置全部适用、无排除，性能组 1189.5 s。eager 平均 0.799189，通过，配置/分段退化 6/33；图平均 1.455079，失败，配置/分段退化 40/204。851 个证据文件校验归档完毕 |
| `wall-lifetime-red.log` / `wall-lifetime-green.log` | 新析构成本回归先复现模拟的 10 μs 调用被旧驱动计为 210 μs；修正后本测试及调用/同步失败传播共 3 项通过，14.61 s |

图 wall 的前一结果现在在下一计时开始前释放，当前调用输出分配与同步仍在计时内；没有修改输入、入口/R、预热或采样次数，没有修补旧样本。生产扩展未变化；检查确认两轮的代表组源码依赖、构建 manifest 与 runtime identity 相同，但完整继承身份不同，因此保留驱动的保守判定并实际重跑。正确性以 `final/summary.json` 为准，最终图/eager 性能分别见 `private-profile/attempt-1792342031198773140/graph-profile/public-profile/{graph,eager}-average.json`。

磁盘普通可用空间为 0 时，root 仍可写保留空间，CANN profiler 却未输出原始记录。实机容器的 `/dev/shm` 仅 64 MB，与 sandbox 的挂载不同，小型探针可通过但完整预检日志会填满。旧证据已逐文件校验后归档至 `wall-profile-initial/`，原目录保留软链接。最终使用 `unshare -Ur -m` 创建任务私有的 4 GB tmpfs；环境包装器 `run_private_storage.py` 在退出时逐文件校验并归档至同一绝对路径，临时挂载已卸载。未修改全局挂载或文件系统设置，采样工具和口径仍在 `tests/mla/`。常规复现应选择有普通可用空间的输出文件系统；使用该包装器重跑时须选择新的输出目录。

同一慢形状的原三段 Init/Attention/Merge 合计约 424 μs；修正任务分配后约 173 μs，融合后约 86 μs，再消除重复准备后约 78 μs，最终复测 77.492 μs。定位 trace 的 Cube 矩阵计算约 2.287 μs（源 2.075 μs），Cube 标量处理约 41.383 μs（源 8.556 μs）；指标不能替代完整任务耗时。最终 TND auto 代表另有 41.381 μs 的 AICPU metadata 生成；prefill FP16 legacy TND 的 attention 为 43.901 μs、metadata+Init 合计 64.881 μs，源完整 Kernel 为 14.000 μs。完整图组的 R=4 开销尤其明显：TND public_4 平均比 2.039383，其他布局 implicit_4 平均比 1.911516。设备标量准备、metadata 生成和多次发射的冗余保留为后续优化方向，本次按用户要求停止图性能优化；不能宣称源 CANN 等速或公开图性能达标。

未保留的实验如实记录：`cache-profile/` 的逐 word 小缓存无收益；`bulk-build*.log` 的错误地址空间转换编译失败；修正 GM 重载后的 `gm-copy-check.log` 第一项报 aicore exception（timeout or trap），已撤掉全部缓存实现及其专属用例。此前裸 pytest 的环境初始化 core 已移到 `unconfigured-parser.core.3`，未留在仓库。旧 sanitizer 曾受 sandbox 的 LSan/ptrace 限制失败，随后外部运行通过；最终仍对冻结版本实际重跑。

上轮完整证据在 `/tmp/mla-m16-recovery/final/attempt-1792327001286361300/`：图专项 233、累计 E2E 861、普通 FA3 504、普通图 2 项通过；40 配置 eager 平均 0.900399，通过，图平均 2.732201，失败。生产源码已变，这些通过不能外推到当前实现，继承旧 E2E 为 0。当前性能集合固定为原图 40 配置加按调用覆盖预选的调度 56 配置，共 96，不按耗时选择输入；正 softcap 仍按 design 单列。

## 环境与复现

Python 3.11.15、torch 2.7.1+cpu、torch_npu 2.7.1、CANN `/usr/local/Ascend/cann-9.0.0-beta.2`，设备 7 是 Ascend910B3（20 AIC/40 AIV）。CATLASS `769cd40a8716b28650b6bebb08db4834eea4462f`；只读 CANN checkout HEAD `2060a3db064c3352fec6262faed8a2decaf3a233`，构建使用固定 `ca18aee610729ca5270a44e2d84c751bc687fbf0` 的 archive。

生产扩展路径为 `flash_attn_npu_3/flash_attn_npu_3.cpython-311-aarch64-linux-gnu.so`，当前 SHA256 为 `886937396456027a850608e240839d22a34f04427b775691597f9679e0bedfa4`，最终以统一驱动的 build/runtime manifest 核验。构建/数学缓存 `/tmp/flash-attention-npu-mla-46928a793876/mla-validation-artifacts`，native `/tmp/mla-m13b/native`。`run.py` 恢复已核验的构建 manifest 环境，避开会话中不能用于实机的 devlib 占位库；该脚本只是临时环境执行器，全部验收实现仍在 `tests/mla/`。

```bash
python3 /tmp/mla-m16-schedule/run.py python3 -m tests.mla --device 7 --suite gqa-options-cpu-check gqa-options-sanitize graph-tools gqa-options-tools --output /tmp/mla-m16-schedule/final-tools
python3 /tmp/mla-m16-schedule/run.py python3 -m tests.mla --device 7 --suite gqa-options-check gqa-options-source gqa-options-boundaries graph-check graph-profile-check gqa-options-profile-check detail-gqa_options detail-graph e2e ordinary graph-ordinary gqa-options-schedule-profile graph-profile --output /tmp/mla-m16-schedule/final
python3 /tmp/mla-m16-schedule/run.py python3 -m tests.mla --device 7 --suite graph-tools --output /tmp/mla-m16-schedule/wall-tools
python3 /tmp/mla-m16-schedule/run.py python3 -m tests.mla --device 7 --suite graph-profile-check gqa-options-check gqa-options-source gqa-options-boundaries graph-check detail-gqa_options detail-graph e2e ordinary graph-ordinary graph-profile --resume --output /tmp/mla-m16-schedule/final
unshare -Ur -m python3 /tmp/mla-m16-schedule/run_private_storage.py /tmp/mla-m16-schedule/private-profile python3 /tmp/mla-m16-schedule/run.py python3 -m tests.mla --device 7 --suite graph-profile --output /tmp/mla-m16-schedule/private-profile
```

本次仅对已完整实测的 M16 图性能差距接受用户明确例外，其余必要检查与证据保持原要求。不更改无头循环 pending/accepted 状态，不重启下一轮；910C 仍未开放。
