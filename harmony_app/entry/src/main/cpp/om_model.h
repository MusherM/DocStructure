#ifndef UNIREC_OM_MODEL_H
#define UNIREC_OM_MODEL_H

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "neural_network_runtime/neural_network_core.h"

struct NativeResourceManager;

class OmModel {
public:
    OmModel() = default;
    ~OmModel();
    OmModel(const OmModel &) = delete;
    OmModel &operator=(const OmModel &) = delete;

    bool Load(NativeResourceManager *manager, const char *rawPath, const char *label,
              const std::vector<std::vector<int32_t>> &inputShapes);
    bool Run();
    void Unload();

    size_t InputCount() const;
    size_t OutputCount() const;
    size_t InputSize(size_t index) const;
    size_t OutputSize(size_t index) const;
    OH_NN_DataType InputType(size_t index) const;
    OH_NN_DataType OutputType(size_t index) const;
    void *InputData(size_t index) const;
    void *OutputData(size_t index) const;
    const std::vector<int32_t> &InputShape(size_t index) const;
    const std::vector<int32_t> &OutputShape(size_t index) const;

private:
    bool CreateTensors(const std::vector<std::vector<int32_t>> &inputShapes);
    static bool SelectNpuDevice(size_t &deviceId);

    std::string label_;
    size_t deviceId_ {0};
    OH_NNExecutor *executor_ {nullptr};
    std::vector<NN_Tensor *> inputs_;
    std::vector<NN_Tensor *> outputs_;
    std::vector<OH_NN_DataType> inputTypes_;
    std::vector<OH_NN_DataType> outputTypes_;
    std::vector<std::vector<int32_t>> inputShapes_;
    std::vector<std::vector<int32_t>> outputShapes_;
};

#endif
