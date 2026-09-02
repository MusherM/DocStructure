#include "om_model.h"

#include "diagnostic_log.h"

#include <CANNKit/hiai_helper.h>
#include <CANNKit/hiai_options.h>
#include <chrono>
#include <cstdlib>
#include <rawfile/raw_file_manager.h>
#include <sstream>

namespace {
using Clock = std::chrono::steady_clock;

std::string ShapeString(const std::vector<int32_t> &shape)
{
    std::ostringstream stream;
    stream << '[';
    for (size_t index = 0; index < shape.size(); ++index) {
        if (index > 0) {
            stream << ',';
        }
        stream << shape[index];
    }
    stream << ']';
    return stream.str();
}

std::vector<int32_t> ReadShape(NN_TensorDesc *description)
{
    std::vector<int32_t> shape;
    int32_t *dimensions = nullptr;
    size_t count = 0;
    if (description != nullptr &&
        OH_NNTensorDesc_GetShape(description, &dimensions, &count) == OH_NN_SUCCESS &&
        dimensions != nullptr) {
        shape.assign(dimensions, dimensions + count);
        std::free(dimensions);
    }
    return shape;
}

OH_NN_DataType ReadType(NN_TensorDesc *description)
{
    OH_NN_DataType type = OH_NN_UNKNOWN;
    if (description != nullptr) {
        OH_NNTensorDesc_GetDataType(description, &type);
    }
    return type;
}

void DestroyTensors(std::vector<NN_Tensor *> &tensors)
{
    for (NN_Tensor *tensor : tensors) {
        OH_NNTensor_Destroy(&tensor);
    }
    tensors.clear();
}

bool ApplyInputShapes(OH_NNCompilation *compilation,
                      const std::vector<std::vector<int32_t>> &inputShapes)
{
    std::vector<NN_TensorDesc *> descriptions;
    descriptions.reserve(inputShapes.size());
    bool ok = true;
    for (const auto &shape : inputShapes) {
        NN_TensorDesc *description = OH_NNTensorDesc_Create();
        if (description == nullptr ||
            OH_NNTensorDesc_SetShape(description, shape.data(), shape.size()) != OH_NN_SUCCESS) {
            ok = false;
            if (description != nullptr) {
                OH_NNTensorDesc_Destroy(&description);
            }
            break;
        }
        descriptions.push_back(description);
    }
    if (ok) {
        ok = HMS_HiAIOptions_SetInputTensorShapes(
            compilation, descriptions.data(), descriptions.size()) == OH_NN_SUCCESS;
    }
    for (NN_TensorDesc *description : descriptions) {
        OH_NNTensorDesc_Destroy(&description);
    }
    return ok;
}
}

OmModel::~OmModel()
{
    Unload();
}

bool OmModel::SelectNpuDevice(size_t &deviceId)
{
    const size_t *deviceIds = nullptr;
    uint32_t deviceCount = 0;
    if (OH_NNDevice_GetAllDevicesID(&deviceIds, &deviceCount) != OH_NN_SUCCESS ||
        deviceIds == nullptr) {
        DiagnosticLog::Fatal("DEVICE", "ENUM_FAILED", "OH_NNDevice_GetAllDevicesID");
        return false;
    }
    DiagnosticLog::Info("DEVICE", "ENUMERATED", "count=" + std::to_string(deviceCount));
    for (uint32_t index = 0; index < deviceCount; ++index) {
        const char *name = nullptr;
        if (OH_NNDevice_GetName(deviceIds[index], &name) != OH_NN_SUCCESS || name == nullptr) {
            DiagnosticLog::Fatal("DEVICE", "NAME_FAILED", std::to_string(index));
            return false;
        }
        DiagnosticLog::Info("DEVICE", "FOUND",
            "id=" + std::to_string(deviceIds[index]) + " name=" + name);
        if (std::string(name) == "HIAI_F") {
            deviceId = deviceIds[index];
            return true;
        }
    }
    DiagnosticLog::Fatal("DEVICE", "NPU_NOT_FOUND", "HIAI_F is unavailable");
    return false;
}

bool OmModel::Load(NativeResourceManager *manager, const char *rawPath, const char *label,
                   const std::vector<std::vector<int32_t>> &inputShapes)
{
    Unload();
    label_ = label;
    DiagnosticLog::Info("ASSET", "OPEN_BEGIN", label_ + " path=" + rawPath);
    RawFile *file = OH_ResourceManager_OpenRawFile(manager, rawPath);
    if (file == nullptr) {
        DiagnosticLog::Fatal("ASSET", "MODEL_OPEN", rawPath);
        return false;
    }
    const long modelSize = OH_ResourceManager_GetRawFileSize(file);
    if (modelSize <= 0) {
        OH_ResourceManager_CloseRawFile(file);
        DiagnosticLog::Fatal("ASSET", "MODEL_SIZE", label_);
        return false;
    }
    std::vector<uint8_t> modelData(static_cast<size_t>(modelSize));
    const int bytesRead = OH_ResourceManager_ReadRawFile(file, modelData.data(), modelData.size());
    OH_ResourceManager_CloseRawFile(file);
    if (bytesRead != modelSize) {
        DiagnosticLog::Fatal("ASSET", "MODEL_READ", label_ + " bytes=" + std::to_string(bytesRead));
        return false;
    }
    DiagnosticLog::Info("ASSET", "OPEN_END", label_ + " bytes=" + std::to_string(modelSize));

    const HiAI_Compatibility compatibility =
        HMS_HiAICompatibility_CheckFromBuffer(modelData.data(), modelData.size());
    DiagnosticLog::Info("MODEL_COMPAT", "CHECKED",
        label_ + " value=" + std::to_string(static_cast<int>(compatibility)));
    if (compatibility != HIAI_COMPATIBILITY_COMPATIBLE) {
        DiagnosticLog::Fatal("MODEL_COMPAT", "INCOMPATIBLE", label_);
        return false;
    }

    OH_NNCompilation *compilation =
        OH_NNCompilation_ConstructWithOfflineModelBuffer(modelData.data(), modelData.size());
    if (compilation == nullptr) {
        DiagnosticLog::Fatal("MODEL_BUILD", "CONSTRUCT_FAILED", label_);
        return false;
    }
    if (!SelectNpuDevice(deviceId_)) {
        OH_NNCompilation_Destroy(&compilation);
        return false;
    }

    bool optionsOk = OH_NNCompilation_SetDevice(compilation, deviceId_) == OH_NN_SUCCESS;
    optionsOk = optionsOk && ApplyInputShapes(compilation, inputShapes);
    optionsOk = optionsOk &&
        HMS_HiAIOptions_SetBandMode(compilation, HIAI_BANDMODE_NORMAL) == OH_NN_SUCCESS;
    std::vector<HiAI_ExecuteDevice> order {HIAI_EXECUTE_DEVICE_NPU};
    optionsOk = optionsOk && HMS_HiAIOptions_SetModelDeviceOrder(
        compilation, order.data(), order.size()) == OH_NN_SUCCESS;
    optionsOk = optionsOk && HMS_HiAIOptions_SetFallbackMode(
        compilation, HIAI_FALLBACK_DISABLED) == OH_NN_SUCCESS;
    optionsOk = optionsOk && HMS_HiAIOptions_SetDeviceMemoryReusePlan(
        compilation, HIAI_DEVICE_MEMORY_REUSE_PLAN_HIGH) == OH_NN_SUCCESS;
    if (!optionsOk) {
        OH_NNCompilation_Destroy(&compilation);
        DiagnosticLog::Fatal("MODEL_BUILD", "OPTION_FAILED", label_);
        return false;
    }
    DiagnosticLog::Info("MODEL_BUILD", "OPTIONS",
        label_ + " device=HIAI_F fallback=disabled memoryReuse=high");
    for (size_t index = 0; index < inputShapes.size(); ++index) {
        DiagnosticLog::Info("MODEL_BUILD", "GEAR_INPUT",
            label_ + " index=" + std::to_string(index) + " shape=" + ShapeString(inputShapes[index]));
    }

    const auto begin = Clock::now();
    const OH_NN_ReturnCode buildResult = OH_NNCompilation_Build(compilation);
    const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now() - begin).count();
    if (buildResult != OH_NN_SUCCESS) {
        OH_NNCompilation_Destroy(&compilation);
        DiagnosticLog::Fatal("MODEL_BUILD", "BUILD_FAILED",
            label_ + " rc=" + std::to_string(buildResult) + " ms=" + std::to_string(elapsed));
        return false;
    }
    executor_ = OH_NNExecutor_Construct(compilation);
    OH_NNCompilation_Destroy(&compilation);
    if (executor_ == nullptr) {
        DiagnosticLog::Fatal("MODEL_BUILD", "EXECUTOR_FAILED", label_);
        return false;
    }
    DiagnosticLog::Info("MODEL_BUILD", "READY", label_ + " ms=" + std::to_string(elapsed));
    return CreateTensors(inputShapes);
}

bool OmModel::CreateTensors(const std::vector<std::vector<int32_t>> &inputShapes)
{
    size_t inputCount = 0;
    size_t outputCount = 0;
    if (OH_NNExecutor_GetInputCount(executor_, &inputCount) != OH_NN_SUCCESS ||
        OH_NNExecutor_GetOutputCount(executor_, &outputCount) != OH_NN_SUCCESS ||
        inputCount != inputShapes.size()) {
        DiagnosticLog::Fatal("IO_CONTRACT", "COUNT_MISMATCH",
            label_ + " expectedInputs=" + std::to_string(inputShapes.size()) +
            " actualInputs=" + std::to_string(inputCount) + " outputs=" + std::to_string(outputCount));
        return false;
    }

    inputTypes_.clear();
    outputTypes_.clear();
    inputShapes_.clear();
    outputShapes_.clear();
    for (size_t index = 0; index < inputCount; ++index) {
        NN_TensorDesc *description = OH_NNExecutor_CreateInputTensorDesc(executor_, index);
        if (description == nullptr) {
            DiagnosticLog::Fatal("IO_CONTRACT", "INPUT_DESC", label_ + " index=" + std::to_string(index));
            return false;
        }
        inputTypes_.push_back(ReadType(description));
        inputShapes_.push_back(ReadShape(description));
        NN_Tensor *tensor = OH_NNTensor_Create(deviceId_, description);
        OH_NNTensorDesc_Destroy(&description);
        if (tensor == nullptr) {
            DiagnosticLog::Fatal("IO_CONTRACT", "INPUT_TENSOR", label_ + " index=" + std::to_string(index));
            return false;
        }
        inputs_.push_back(tensor);
        DiagnosticLog::Info("IO_CONTRACT", "INPUT",
            label_ + " index=" + std::to_string(index) + " type=" +
            std::to_string(inputTypes_.back()) + " shape=" + ShapeString(inputShapes_.back()) +
            " bytes=" + std::to_string(InputSize(index)));
    }
    for (size_t index = 0; index < outputCount; ++index) {
        NN_TensorDesc *description = OH_NNExecutor_CreateOutputTensorDesc(executor_, index);
        if (description == nullptr) {
            DiagnosticLog::Fatal("IO_CONTRACT", "OUTPUT_DESC", label_ + " index=" + std::to_string(index));
            return false;
        }
        outputTypes_.push_back(ReadType(description));
        outputShapes_.push_back(ReadShape(description));
        NN_Tensor *tensor = OH_NNTensor_Create(deviceId_, description);
        OH_NNTensorDesc_Destroy(&description);
        if (tensor == nullptr) {
            DiagnosticLog::Fatal("IO_CONTRACT", "OUTPUT_TENSOR", label_ + " index=" + std::to_string(index));
            return false;
        }
        outputs_.push_back(tensor);
        DiagnosticLog::Info("IO_CONTRACT", "OUTPUT",
            label_ + " index=" + std::to_string(index) + " type=" +
            std::to_string(outputTypes_.back()) + " shape=" + ShapeString(outputShapes_.back()) +
            " bytes=" + std::to_string(OutputSize(index)));
    }
    return true;
}

bool OmModel::Run()
{
    if (executor_ == nullptr || inputs_.empty() || outputs_.empty()) {
        DiagnosticLog::Fatal("INFERENCE", "MODEL_NOT_READY", label_);
        return false;
    }
    const OH_NN_ReturnCode result = OH_NNExecutor_RunSync(
        executor_, inputs_.data(), inputs_.size(), outputs_.data(), outputs_.size());
    if (result != OH_NN_SUCCESS) {
        DiagnosticLog::Fatal("INFERENCE", "RUN_FAILED", label_ + " rc=" + std::to_string(result));
        return false;
    }
    return true;
}

void OmModel::Unload()
{
    DestroyTensors(inputs_);
    DestroyTensors(outputs_);
    if (executor_ != nullptr) {
        OH_NNExecutor_Destroy(&executor_);
        executor_ = nullptr;
    }
    inputTypes_.clear();
    outputTypes_.clear();
    inputShapes_.clear();
    outputShapes_.clear();
    deviceId_ = 0;
}

size_t OmModel::InputCount() const { return inputs_.size(); }
size_t OmModel::OutputCount() const { return outputs_.size(); }

size_t OmModel::InputSize(size_t index) const
{
    size_t size = 0;
    return index < inputs_.size() && OH_NNTensor_GetSize(inputs_[index], &size) == OH_NN_SUCCESS ? size : 0;
}

size_t OmModel::OutputSize(size_t index) const
{
    size_t size = 0;
    return index < outputs_.size() && OH_NNTensor_GetSize(outputs_[index], &size) == OH_NN_SUCCESS ? size : 0;
}

OH_NN_DataType OmModel::InputType(size_t index) const
{
    return index < inputTypes_.size() ? inputTypes_[index] : OH_NN_UNKNOWN;
}

OH_NN_DataType OmModel::OutputType(size_t index) const
{
    return index < outputTypes_.size() ? outputTypes_[index] : OH_NN_UNKNOWN;
}

void *OmModel::InputData(size_t index) const
{
    return index < inputs_.size() ? OH_NNTensor_GetDataBuffer(inputs_[index]) : nullptr;
}

void *OmModel::OutputData(size_t index) const
{
    return index < outputs_.size() ? OH_NNTensor_GetDataBuffer(outputs_[index]) : nullptr;
}

const std::vector<int32_t> &OmModel::InputShape(size_t index) const { return inputShapes_.at(index); }
const std::vector<int32_t> &OmModel::OutputShape(size_t index) const { return outputShapes_.at(index); }
