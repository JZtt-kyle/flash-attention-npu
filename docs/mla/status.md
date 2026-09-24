# MLA 当前交接

## 状态与下一步

- 当前步骤：M02a 已完成；CANN 执行器、910B 精度/调用计时、普通 FA3 强制重建回归及本步质量检查通过。
- 当前分支：`feat/mla-cann-migration`；接手基点 M01 `d40f0a8`，源码与交接一致。
- 用户已授权按现有方案继续；已确认范围内不重复审批。M02 在实现前拆为 M02a/M02b：安装包溯源发现版本差异，执行器验收与固定源码性能基线分开闭环。
- 本轮产物：[CANN 执行器](../../tests/mla_cann_baseline.py)、[910B 验收](../../tests/test_flash_attn_npu_v3_mla_cann.py)，以及三份职责文档更新；本次闭环提交由 `git log -1 -- docs/mla/status.md` 定位。
- 唯一下一步：M02b，在源仓外建立固定 CANN commit 的 FIA 基线，并用 profiler 确认实际模板、kernel 时间和转换成本。本轮一个提交后停止。

本步只验收测试用 CANN 执行器。**生产 MLA kernel/API 尚未接通；安装包计时不是固定源码性能基线，也没有得出迁移性能通过的结论。**

## 最近闭环的证据

| 检查 | 结果 |
| --- | --- |
| M01 接手复核 | PASS：工作区仅有既有未跟踪 `atk_test/`；CPU reference 重跑 `71 passed in 21.15s` |
| CANN 设备精度 | PASS：最终 `12 passed in 21.18s`，无 skip；D01 默认/0.23/0 scale、D02 多 query、P01 split/combined，均覆盖 FP16/BF16；独立 O/LSE 双 reference、重复执行和全部内容/RoPE 输入不被改写通过 |
| 精度口径 | 沿用原仓 `assert_fa_close`，未修改 comparator 或 reference，未放宽阈值；两轮独立执行器运行的逐例误差完全一致；最大 O 误差 FP16 `0.001953125`、BF16 `0.0078125`，最大 LSE 误差 `5.722046e-6` |
| prefill 输出形状 | 初次默认包装器的 combined 192/128 调用失败：错误分配 O 末维 192，CANN 要求 128。执行器统一使用显式 O/LSE 的 `.out` 重载，两种布局精度通过；split 视图保留行 stride 192、RoPE offset 128；初次失败日志保留 |
| 调用计时 | PASS：12 组输入各执行两轮独立命令，每轮预热 10、5 轮 × 50 次 event span 及逐调用同步 wall 计时；完整原始样本和输入/环境已保存。event 含 host 发射间隙及可能的转换，不称为 kernel 时间 |
| 普通 FA3 重建 | PASS：仅 v3/910，`FLASH_ATTN_FORCE_REBUILD=TRUE`，9 个 ASC 编译单元、AICPU 编译、链接及 inplace 复制成功，退出码 0；实际加载路径与新哈希已核对 |
| 普通 FA3 回归 | PASS：重建后 `8 passed in 25.72s`，无 skip；dense、TND、paged cache、FP16/BF16 split-KV 和 metadata 覆盖，精确节点见下方命令 |
| 溯源边界 | 安装包为 CANN `9.0.0-beta.2`，ops-transformer 安装时间戳 `20260328_213255003`；已记录进程实际加载的 opapi/host/tiling 库。安装的 4 个 MLA/通用非量化 header 哈希均不同于固定源码；二进制对应关系仍未建立，M02b 必须处理 |
| 仓库质量 | PASS：`ci/code_quality.sh --staged` 检查本步五个文件；空白、末尾换行、冲突标记、Ruff check/format 均通过，其余语言 hook 无适用文件；文档链接/围栏与 `git diff --check` 通过 |

持久证据位于本仓专用忽略目录 `/home/zhangqijia/repo/flash-attention-npu/build/mla-m02-validation/`，不加入 Git：

- `m01-reference.log`、`cann-pytest.log`、`cann-pytest-final.log`、`build.log`、`fa3-regression.log`、`fa3-extension.json`、`npu-smi-before-regression.log`。
- `cann-baseline.json` / `.log` 与 `cann-baseline-final.json` / `.log`：每例原始计时、精度阈值/误差、用例 ID、dtype、shape/stride、scale、split/metadata 口径、库路径与哈希；final 另含 storage offset 和最终执行器哈希，已核对与当前文件一致。
- `probe.log` 保留 combined 默认包装器失败，`probe-out.log` 记录显式输出探测成功；探测脚本为忽略的临时文件，复现使用仓内正式执行器。
- `quality.log`、`quality-final.log` 记录仓库质量检查，`checksums.sha256` 固定主要验收证据文件的哈希。

最终一轮调用诊断值（FP16 / BF16，微秒）：D01 默认 event `103.37 / 101.99`、同步 wall `163.61 / 160.66`；P01 split event `205.44 / 200.47`、wall `286.60 / 274.83`；P01 combined event `112.70 / 111.98`、wall `184.68 / 184.16`。这些是含包装器成本的调用时间；两轮之间存在 host 噪声，后续性能验收仍需固定源码、profiler 和双方交替测量。

## 最小闭环复现

从仓库根执行。设备需先重新确认健康空闲；下方 7 为本轮物理设备，进程内为 `npu:0`。设备命令本轮在沙箱外执行。

```bash
TORCH_DEVICE_BACKEND_AUTOLOAD=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -m pytest -q tests/test_flash_attn_npu_v3_mla_reference.py

ASCEND_RT_VISIBLE_DEVICES=7 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -m pytest -q tests/test_flash_attn_npu_v3_mla_cann.py

ASCEND_RT_VISIBLE_DEVICES=7 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -m tests.mla_cann_baseline \
  --output build/mla-m02-validation/reproduce.json \
  --cann-source /home/zhangqijia/repo/ops-transformer \
  --seed 20260924 --warmup 10 --iterations 50 --rounds 5
```

执行器拒绝覆盖已有 JSON；复现时选择新的输出路径。默认跑全部 12 组，`--cases` / `--dtypes` 可缩小范围；不支持的设备直接失败，不用 skip 充当通过。`TORCH_DEVICE_BACKEND_AUTOLOAD=0 python -m tests.mla_cann_baseline --help` 无需设备。

重建与普通 FA3 最小回归：

```bash
TORCH_DEVICE_BACKEND_AUTOLOAD=0 FLASH_ATTN_BUILD_VERSION=v3 \
  FLASH_ATTN_BUILD_NPU=910 FLASH_ATTN_SKIP_SUBMODULE_INIT=1 \
  FLASH_ATTN_FORCE_REBUILD=TRUE python setup.py build_ext --inplace

ASCEND_RT_VISIBLE_DEVICES=7 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -m pytest -q \
  tests/test_flash_attn_npu_v3_metadata.py::test_flash_attn_func_metadata_bsnd \
  'tests/test_flash_attn_npu_v3_metadata.py::test_flash_attn_varlen_func_metadata_tnd[data_type0-1-1-1-512-1024-128-True]' \
  'tests/test_flash_attn_npu_v3_metadata.py::test_flash_attn_varlen_func_metadata_tnd[data_type2-7-5-1-512-512-128-True]' \
  'tests/test_flash_attn_npu_v3_metadata.py::test_flash_attn_kvcache_metadata_bsnd[data_type0-1-1-1-1024-1024-128-128-False]' \
  'tests/test_flash_attn_npu_v3_metadata.py::test_flash_attn_kvcache_metadata_flash_decode[data_type1-8-1-64-2-False-0.0-0]' \
  'tests/test_flash_attn_npu_v3_metadata.py::test_flash_attn_kvcache_metadata_flash_decode[data_type2-8-1-64-2-False-0.0-0]'
```

质量入口（复用 M01 的固定版本临时工具，干净环境按 `ci/quality-requirements.txt` 安装）：

```bash
PYTHONPATH=/tmp/mla-m01-precommit PRE_COMMIT_HOME=/tmp/mla-m01-precommit-cache \
  ci/code_quality.sh --staged
git diff --check
git diff --cached --check
```

## 环境与下轮约束

| 项目 | 本轮核实结果（2026-09-24） |
| --- | --- |
| CANN 源仓 / CATLASS | 分别为 `ca18aee610729ca5270a44e2d84c751bc687fbf0` / `769cd40a8716b28650b6bebb08db4834eea4462f`；FIA 与 attention/common scoped status、CATLASS status 均无改动，继续只读 |
| Python / 包 | `/usr/local/python3.11.15/bin/python`，Python 3.11.15；实际 torch `2.7.1+cpu`、torch_npu `2.7.1` 均导入并运行 |
| CANN / OPP | `/usr/local/Ascend/cann-9.0.0-beta.2` 及其 `opp`；未设 `ASCEND_CUSTOM_OPP_PATH`；完整环境与实际加载库见 final JSON |
| 实际 FIA 库 | `aarch64-linux/lib64/libopapi_transformer.so` SHA256 `4a16b14a04299ea846268deb6f5e2b5e7fb8df6005dcf71ad63ea7cbc7b546d0`；OPP `op_host/lib/linux/aarch64/libophost_transformer.so` SHA256 `9715d79dc8a008256a3ba830b0b47b4cea8d68e1b91ff6802d865348b7e071f8`；另记录 `libopapi.so` 与 `liboptiling.so` |
| 重建 FA3 扩展 | 实际加载本仓 `flash_attn_npu_3/flash_attn_npu_3.cpython-311-aarch64-linux-gnu.so`；SHA256 `fabdbb711f336e2a062e0b032fe0bdb65758da091b61c6537a0b8294a6c2677c`，已替代上轮未经构建证明的旧产物 |
| 设备 | 物理 7 号 Ascend910B3，运行前健康且无进程；5、6 号仍 Alarm，未使用；后续重新查询再选卡 |
| 沙箱 / shell | 延续上轮已确认的设备访问差异，实机命令在沙箱外运行；使用 `login=false` 避免 login 初始化阻塞；重复 tiling 注册 warning 仍存在，所有验收进程退出码为 0 |
| 既有文件 | `atk_test/` 为接手前既有未跟踪目录，未改、未暂存；大日志和构建产物均在忽略目录 |

M02b 必须先解决安装包与固定源版本的对应关系，不能直接把本轮 event span 当作原 CANN kernel 基线。固定源的构建应放源仓之外，保留 commit、构建命令、加载路径/哈希及 profiler 原始数据。普通 FA3 已建立重建后的最小回归入口；源码、环境或构建变化时重跑受影响检查。

TND/PA、mask、split/metadata 和图的 MLA 适配尚未接通。现生产入口的 qv 静默忽略缺口仍留给 M03，不能将当前调用成功视为 MLA 迁移完成。
