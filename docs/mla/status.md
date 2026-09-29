# MLA 当前状态

## 当前能力与下一步

- 分支：`feat/mla-cann-migration`；接手 HEAD 为 `8b7e08a`，接手工作区干净。本次将 M05b2 拆为签名兼容（M05b2a）和公开生成/消费（M05b2b），完成 M05b2a。
- 公开 `get_scheduler_metadata` 的 21 个位置参数及默认值与固定官方 FA3 一致，使用 `has_softcap`。内部 custom op、fake 和 C++ 参数顺序已同步；额外 `softcap` / `softmax_scale` 仅保留为公开关键字兼容项。
- 普通 FA3 仍返回连续、2384 字节的 `uint8` metadata，保存原有 softcap/scale 数值。旧正 `softcap` 关键字调用继续有效；仅提供 `has_softcap=True` 时明确拒绝，须同时提供正的旧 softcap 数值；True 与零值冲突、负值或非有限 softcap 均拒绝。兼容规则见 [design 第 5 节](design.md#5-metadata-设计)。
- MLA 能力未扩展：公开 dense 仍为 FP16/BF16、512+64 共享 KV、连续 BSND、B>0、`1<=Sq*Hq<=256`、KV `1..512`、Hkv=1、无 mask、显式 `num_splits=1`。AICPU → 401 字 int32 → dense Kernel 的链路仍只在独立测试中接通；公开 MLA metadata 在真实/fake/C++ 中均明确拒绝。
- 唯一下一步：M05b2b，接通该 dense 范围的公开 metadata 生成和消费，包括必需的 `cache_seqlens` 设备读取与范围拒绝、设备校验错误上报、运行时 scale、allocator/stream 生命周期与有/无 metadata 复用验收。M05c 再验收最终 E2E 门槛；整个 M05 完成前不开始 M04c1。本步提交后停止。

## 最近一次闭环证据

证据目录：`build/mla-m05b2-public/`；物理 7 号 Ascend910B3。最终共 **771 项 pytest 检查通过，无失败或 skip**。

| 项目 | 结果 |
| --- | --- |
| 上步复核 | PASS：M05b1 的 2,620 份证据哈希、958 项通过结果、已验收源码、软件栈、实际扩展及固定源仓状态一致 |
| 官方签名 | PASS：对固定提交源码的 AST 逐项核对 21 个位置参数及默认值，额外两项为 keyword-only；源码副本和哈希在 `upstream/`、`upstream-reference.json`、`official-signature.json` |
| 新增签名与兼容 | PASS：129 项；公开/custom op/C++ 与 fake 的有效和拒绝路径、旧调用 fingerprint、24 组双 dtype O/LSE 精度，以及 4 组更新长度后的 torch.compile 检查 |
| 旧协议兼容 | PASS：改动前后 80 组 metadata 逐字节一致；双 dtype、dense/TND/分页、无 mask/causal/window、split、softcap 0/15 和 scale 默认/0/0.23/-0.11 |
| MLA 回归 | PASS：622 项，含 600 项真实/fake/C++ 契约和 22 项公开精度、舍入、交错 scale/stream 与输入生命周期检查；精度阈值未调整 |
| 普通 FA3 回归 | PASS：20 项，覆盖 BSND、TND、分页、split、metadata、mask、softcap/scale、错误拒绝及编译图前反向 |
| 构建与加载 | PASS：增量重编译 `flash_api.cpp`，实际链接 17 个对象；其余 14 个 ASC 和 2 个 AICPU 对象哈希不变。加载扩展与构建产物一致，源码和依赖已核查 |
| 提交质量 | 六项适用仓库质量检查、文档链接、差异和证据完整性检查通过；见 `quality-staged.log`、`acceptance.json` 和 `checksums.sha256` |

本步未修改调度算法、AICPU、Kernel/Block/Tile 或 launcher，未重跑未受影响的独立 CPU/sanitizer 矩阵及 Kernel 性能采样。M05b1 的设备消费者精度、错误拒绝及性能证据保留在 `build/mla-m05b1-consumer/`。本步的前向回归重新构建双 dtype 固定源/迁移探针，但不据此声称有新的性能结果。

**公开 MLA metadata 链路及 M05 整体尚未完成，最终性能门槛未达成。** M05b2b 接通后仍须在 M05c 按 [design](design.md) 逐例验收 `FA3 metadata E2E / 同轮原 CANN E2E <= 1.0`，分开记录生成、复用和完整调用段。

## 环境、失败与限制

- Python `/usr/local/python3.11.15/bin/python3` 3.11.15；torch `2.7.1+cpu`，torch_npu `2.7.1`；CANN `/usr/local/Ascend/cann-9.0.0-beta.2`，bisheng clang 15，`dav-2201`、`-O2`。
- 当前实际扩展为本仓 `flash_attn_npu_3/flash_attn_npu_3.cpython-311-aarch64-linux-gnu.so`，SHA256 `713322ade135a6b99ce0dec7f466823d3bed805c8050538875c35638a3074445`。新扩展及 `flash_api.o` 副本保存在 `production-artifacts/`；61 个源码文件、17 个对象与依赖证据见 `build-source-manifest.json`、`build-provenance.json`、`production-snapshots.json`。
- 7 号 910B3 开始和结束健康为 OK、ECC 为 0，结束无残留设备进程；见 `environment.json`、`environment-after.json`。NPU 命令在沙箱外运行；登录 shell 最初长时间无输出，后续使用非登录 shell。
- 固定 CANN `ca18aee610729ca5270a44e2d84c751bc687fbf0`、CATLASS `769cd40a8716b28650b6bebb08db4834eea4462f`；CANN 源仓既有删除/未跟踪文件保持原样。固定官方 FA3 为 `d15f1531a460ba456f41b01a774f33ab2db8febf`，本步使用已有缓存源码核对签名并保存副本。
- 本步只迁移 910 后端的 metadata 签名，不改变 950 后端。旧数值参数的位置调用不再属于兼容接口，应改用关键字。MLA metadata 无返回占位结果，也未调用原 CANN 代替迁移。
- 本轮无构建或测试失败。首次 clang-format 自动调整 C++ 换行后复查通过；CANN beta 版本和重复 tiling 注册警告保留在日志。整理证据时首次脚本错误地要求测试结束后仍存在 `kernel_meta/`，实际已不存在，随后按存在时归档、缺失时确认无需清理修正；见 `artifact-cleanup.json`。提交前根目录无该临时目录。

## 证据与复现

- `previous-verified.json`：上步证据、源码及源仓状态；`environment*`：实际加载扩展和设备条件。
- `build-final*`、`build-source-manifest.json`、`build-provenance*`、`production-artifacts/`：实际重编译、链接、依赖、哈希与产物副本。
- `byte-metadata-before.json`、`byte-metadata-after.json`、`compare_byte_metadata.py`：旧/新扩展哈希、全部 80 组输入参数与 metadata 原始字节。
- `signature-final*`、`accuracy-final*`、`regression-final*`：精确命令、环境、退出码、原始日志、JUnit 和双 dtype 精度/构建证据。`regression-nodes.json`、`run_regressions.py` 保存本轮节点选择及顺序。

从仓库根目录复现，使用空闲健康设备并替换 artifact/JUnit/日志路径以保留既有证据：

```bash
FLASH_ATTN_BUILD_VERSION=v3 FLASH_ATTN_BUILD_NPU=910 FLASH_ATTN_SKIP_SUBMODULE_INIT=1 python3 setup.py build_ext --inplace
ASCEND_RT_VISIBLE_DEVICES=7 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 -m pytest -x -q tests/test_flash_attn_npu_v3_metadata_signature.py tests/test_flash_attn_npu_v3_mla_contract.py
```

其余公开 MLA 精度/生命周期及普通 FA3 验收按 `accuracy-final-command.json`、`regression-final-command.json` 的命令和环境复验。旧/新 byte 对照的 before 数据由接手扩展生成，不能用新扩展重写；复核时比较保存的原始字节与当前生成结果。

本步提交前检查：

```bash
PYTHONPATH=/tmp/mla-m01-precommit PRE_COMMIT_HOME=/tmp/mla-m01-precommit-cache bash ci/code_quality.sh --staged
python3 build/mla-m05b2-public/verify_acceptance.py --staged
git diff --check
git diff --cached --check
```

当前闭环提交用 `git log -1 -- docs/mla/status.md` 定位。
