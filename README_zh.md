# UniRec-0.1B 动态档位 OM Demo

English: [README.md](README.md)

本仓库包含两个交付部分：

1. 可复现的 UniRec-0.1B 导出链路：生成一份动态档位 encoder ONNX 和一份动态档位 decoder ONNX，验证全部四档，并提供在 Linux 上生成一份动态档位 encoder OM 与一份动态档位 decoder OM 的 OMG 脚本。
2. 原生 HarmonyOS App：通过 CANN Kit/NNRt 加载这两个 OM，为输入图片选择档位，执行 encoder + 自回归 decoder 推理，并在端侧解码 token。

本仓库不会声称 OM 已完成转换：你必须在 Linux 上使用与目标手机精确匹配的 CANN Kit/工具链手动完成。OMG 转换成功并不能证明 OM 与手机兼容，也不能证明模型能完整运行在手机 NPU 上。

四档实测结果与仍需完成的端侧验证边界见 [交付与验证报告](dev/delivery_report.html)。

## 模型契约

需求中的尺寸按**宽×高**解释。运行时张量采用 NCHW，因此 shape 为：

| 档位 | Encoder 输入 | 视觉 token 数 | Encoder `cross_k/cross_v` |
|---|---:|---:|---:|
| 384×512 | `[1,3,512,384]` | 192 | `[6,1,6,192,128]` |
| 576×768 | `[1,3,768,576]` | 432 | `[6,1,6,432,128]` |
| 768×1024 | `[1,3,1024,768]` | 768 | `[6,1,6,768,128]` |
| 960×1408 | `[1,3,1408,960]` | 1320 | `[6,1,6,1320,128]` |

整个 Demo 只有一个 encoder 模型和一个 decoder 模型。Decoder 包含四个关联的视觉序列档位，自注意力 cache 容量固定为 2048 token：

```text
past_keys / past_values: [6,1,6,2048,128]
new_keys  / new_values:  [6,1,6,1,128]
logits:                    [1,56371]
```

固定容量 cache 避免每生成一个 token 就引入一个新的动态 shape。App 每步只更新当前 cache 槽位。代价是每步自注意力都要对带 mask 的 2048 槽 cache 计算；这种方式更容易转为 OM，但可能比真正的可变长度 cache 更慢。

图片使用双三次插值等比缩放，左上对齐，右侧和底部补白；RGB 归一化公式为 `(x / 255 - 0.5) / 0.5`。对给定的超宽验证图，必须保持左上对齐；居中补白会导致三个档位出现错误的开头 token。

## 第一部分：导出 ONNX 并转换为 OM

### 1. 前置条件

- Python 3.10–3.12 与 [uv](https://docs.astral.sh/uv/)。
- 官方 [Topdu/OpenOCR](https://github.com/Topdu/OpenOCR) 仓库。本 Demo 在提交 `0d522801ec6dc1df852c6b6d4ed6a08f5127ed97` 上完成验证。
- 将官方 [topdu/unirec-0.1b](https://huggingface.co/topdu/unirec-0.1b) 文件放在同一个本地目录：

```text
unirec-0.1b/
├── config.json
├── generation_config.json
├── model.pth
├── special_tokens_map.json
├── tokenizer.json
└── tokenizer_config.json
```

导出器使用 `torch.load(..., weights_only=True)` 安全读取 `model.pth`，并拒绝任何缺失或多余的 state-dict 键。

只安装一次 Python 依赖：

```bash
uv sync
```

模型与源码准备示例：

```bash
git clone https://github.com/Topdu/OpenOCR.git third_party/OpenOCR
git -C third_party/OpenOCR checkout 0d522801ec6dc1df852c6b6d4ed6a08f5127ed97
uv run hf download topdu/unirec-0.1b \
  model.pth config.json generation_config.json tokenizer.json \
  tokenizer_config.json special_tokens_map.json \
  --local-dir models/unirec-0.1b
```

### 2. 导出改造后的 ONNX

```bash
./scripts/export_onnx.sh \
  third_party/OpenOCR \
  models/unirec-0.1b \
  model_tools/output
```

生成文件：

```text
model_tools/output/
├── unirec_encoder.onnx
├── unirec_decoder.onnx
├── unirec_tokenizer.bin
├── model_contract.json
└── export_onnx.log
```

Encoder 会预计算 decoder 六层 cross-attention 的 K/V。Decoder 将 12 个分散的可变长度 cache 输入合并为两个固定容量张量，并且只输出单步 K/V 增量。这个改造在不改变训练权重的前提下降低动态 shape 压力与主机/设备拷贝量。

### 3. 验证四档 ONNX

仓库内置验证图为 [model_tools/assets/verify.png](model_tools/assets/verify.png)。成功标准是每个被选档位的输出都以 `已处理` 开头。

```bash
./scripts/verify_onnx.sh \
  model_tools/output \
  model_tools/assets/verify.png \
  已处理
```

验证器会用 ONNX Checker 检查两个模型、检查每一档 K/V shape、使用 ONNX Runtime 执行 encoder + 固定 cache decoder，并写出 `verify_report.json`。本机已验证结果：

| 档位 | 结果 | Token ID |
|---|---|---|
| 384×512 | `已处理` | `0, 23709, 11536` |
| 576×768 | `已处理` | `0, 23709, 11536` |
| 768×1024 | `已处理` | `0, 23709, 11536` |
| 960×1408 | `已处理` | `0, 23709, 11536` |

### 4. 在 Linux 上转换为一对动态档位 OM

必须使用与目标 HarmonyOS 设备匹配的 CANN Kit OMG 转换工具链。不要猜测 `PLATFORM`，也不要因为 OMG 能接受参数就拿其他平台代替目标平台。先加载工具链环境，再运行：

```bash
source /path/to/cann/set_env.sh
PLATFORM='<CANN Kit 包/目标设备文档给出的精确目标平台>' \
  ./scripts/convert_to_om.sh model_tools/output om_output
```

脚本只生成：

```text
om_output/
├── unirec_encoder.om      # 四个 H/W 档位
├── unirec_decoder.om      # 四个关联视觉 token 档位
├── omg_encoder.log
├── omg_decoder.log
└── SHA256SUMS
```

Encoder 动态维度为 `512,384;768,576;1024,768;1408,960`。Decoder 针对 `cross_k` 和 `cross_v` 的动态维度对为 `192,192;432,432;768,768;1320,1320`。浮点模型输入/输出编译为 FP16 以减少内存流量，token 与位置输入保持 INT32。

脚本默认调用 `omg`；如果可执行文件不在 `PATH` 中，可通过 `OMG_BIN` 指定。脚本使用 OMG 面向 ONNX 的 `--input_type` 和 `--output_type` 映射；由于 `--input_fp16_nodes` 对 ONNX 不生效，脚本不会使用该参数。通用 Ascend Toolkit 生成的 OM 不会自动成为可用于手机 CANN Kit 的 OM。

## 第二部分：原生 HarmonyOS App

### 仓库之外必须放入的文件

打开/构建 `harmony_app` 前，将 Linux 生成的两个模型按以下精确路径和名称复制：

```text
harmony_app/entry/src/main/resources/rawfile/models/unirec_encoder.om
harmony_app/entry/src/main/resources/rawfile/models/unirec_decoder.om
```

仓库已经包含：

```text
harmony_app/entry/src/main/resources/rawfile/models/unirec_tokenizer.bin
harmony_app/entry/src/main/resources/rawfile/images/verify.png
```

如果重新生成了 ONNX/tokenizer 产物，请替换 App 中对应 tokenizer：

```bash
cp model_tools/output/unirec_tokenizer.bin \
  harmony_app/entry/src/main/resources/rawfile/models/unirec_tokenizer.bin
```

不要混用不同导出批次的 encoder、decoder 和 tokenizer。原生 App 会强制检查输入/输出数量、相关 shape、dtype、词表大小与特殊 token ID。

### 构建与运行

工程采用华为官方 [CANN Kit C++ 示例](https://gitee.com/harmonyos_samples/cannkit-samplecode-clientdemo-cpp)相同的集成链路，链接 `hiai_foundation`、`libneural_network_core.so`、N-API、HiLog 与 RawFile。

建议基线：

- DevEco Studio 6.0.0 Release 或更高版本。
- HarmonyOS SDK 6.0.0 Release 或更高版本。
- 目标华为手机/平板运行 HarmonyOS 5.1.0 Release 或更高版本，并暴露 `HIAI_F` 设备。
- API 18 构建目标（所用 NNRt/CANN API 的起始版本更早，但当前官方示例采用这一基线）。

使用 DevEco Studio 打开 `harmony_app`，配置签名，为 `arm64-v8a` 构建 `entry` HAP，安装到目标设备，然后点击“运行内置验证图”或“选择图片”。

App 始终只使用一对动态 OM。它根据源图片像素面积选择最小档位，等比缩放，在 `OH_NNCompilation_Build` 前通过 `HMS_HiAIOptions_SetInputTensorShapes` 选择该档；只有档位改变时才重建 executor。

### NPU-only 策略

App 显式设置：

```text
HMS_HiAIOptions_SetModelDeviceOrder(... NPU ...)
HMS_HiAIOptions_SetFallbackMode(... DISABLED)
```

不会发生静默 CPU 回退。不支持的算子、不兼容 OM、缺少 NPU 设备、Build 失败和 I/O 契约不一致都会成为致命错误。由于两个模型与 K/V buffer 较大，启用了高内存复用方案。

### 异步诊断与致命日志终止

推理以原生异步 Promise 暴露给 ArkTS，因此模型 Build 和同步 NNRt 执行不会阻塞 UI 事件处理函数。原生日志 tag 为 `UniRecOM`，采用包含单调递增 `seq`、`stage`、`event`、`detail` 的类 JSON 记录。

阶段包括：

```text
BOOT → TOKENIZER → ASSET → MODEL_COMPAT → DEVICE → MODEL_BUILD
→ IO_CONTRACT → PREPROCESS → ENCODER → DECODER → POSTPROCESS → DONE
```

Decoder 日志只记录前四步、每第 25 步和 EOS，不会输出完整张量。第一次致命错误只会产生一条包含 `"terminal":true` 的记录；随后原子熔断器会禁止该进程继续输出任何原生业务日志。资源清理仍会静默执行。下一次异步验证前必须重启 App。

日志采集示例：

```bash
hdc shell hilog | grep UniRecOM
```

应保留从进程启动到 `DONE/SUCCESS`，或到唯一一条终止性 `FATAL` 为止的完整日志。

## 重要限制

- 本仓库证明的是改造后 ONNX 在 CPU ONNX Runtime 上正确。这里不声称 OM 转换或手机运行已经成功；它们需要你的 Linux CANN 工具链和物理目标设备。
- OMG 最多支持 16 个动态 shape 档位，转换能否成功仍取决于计算图与目标编译器；本项目使用 4 档。
- OMG 成功不等于手机兼容。App 会执行 `HMS_HiAICompatibility_CheckFromBuffer`、要求 `HIAI_F`、关闭回退，然后要求 `OH_NNCompilation_Build` 与所有 I/O 契约检查全部通过。
- OM 与硬件/工具链强绑定。为昇腾服务器/加速卡编译的模型通常不能假定可在麒麟手机 NPU 上运行。
- FP16 编译可能改变边界 token 的 logits。应在设备上重新运行内置图片并比较开头。如果确有需要，可以设置 `FLOAT_TYPE=FP32`，但需要接受更高内存与 NPU 覆盖下降的风险。
- 模型与运行时的组合内存占用很大。HAP 打包限制、设备 RAM、连续共享内存分配和热限制仍可能阻止执行。
- NNRt/CANN Kit 只能运行目标驱动实现的算子。模型可能成功转换却在兼容检查/Build 失败，也可能成功 Build 后在运行时失败。
- 动态多档不等于任意动态 shape；只能接受声明的四个档位。

## 仓库结构

```text
model_tools/                 Python 导出、预处理、tokenizer、验证
scripts/export_onnx.sh       可复现 ONNX 导出封装
scripts/verify_onnx.sh       四档 ONNX 正确性检查
scripts/convert_to_om.sh     Linux 手工 OMG 动态档位转换
harmony_app/                 原生 HarmonyOS CANN Kit/NNRt App
dev/                         交付与验证报告
```

## 参考资料

- [UniRec-0.1B 模型与论文链接](https://github.com/Topdu/OpenOCR/blob/main/docs/unirec.md)
- [官方 UniRec ONNX 推理实现](https://github.com/Topdu/OpenOCR/blob/main/tools/infer_unirec_onnx.py)
- [HarmonyOS Neural Network Runtime 介绍](https://gitee.com/openharmony/docs/blob/master/en/application-dev/ai/nnrt/Neural-Network-Runtime-Kit-Introduction.md)
- [华为 CANN Kit 模型转换](https://developer.huawei.com/consumer/en/doc/harmonyos-guides/cannkit-model-conversion)
- [OMG 参数文档](https://developer.huawei.com/consumer/en/doc/hiai-guides/overall-parameter-0000001052966900)
