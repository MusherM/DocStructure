#include "unirec_engine.h"

#include "diagnostic_log.h"
#include "model_contract.h"

#include <CANNKit/hiai_helper.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <limits>
#include <sstream>

namespace {
using Clock = std::chrono::steady_clock;

uint16_t FloatToHalf(float value)
{
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    const uint32_t sign = (bits >> 16U) & 0x8000U;
    int32_t exponent = static_cast<int32_t>((bits >> 23U) & 0xFFU) - 127 + 15;
    uint32_t mantissa = bits & 0x7FFFFFU;
    if (exponent <= 0) {
        if (exponent < -10) {
            return static_cast<uint16_t>(sign);
        }
        mantissa = (mantissa | 0x800000U) >> static_cast<uint32_t>(1 - exponent);
        return static_cast<uint16_t>(sign | ((mantissa + 0x1000U) >> 13U));
    }
    if (exponent >= 31) {
        return static_cast<uint16_t>(sign | 0x7C00U);
    }
    return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exponent) << 10U) |
        ((mantissa + 0x1000U) >> 13U));
}

float HalfToFloat(uint16_t value)
{
    const uint32_t sign = (static_cast<uint32_t>(value & 0x8000U)) << 16U;
    const uint32_t exponent = (value >> 10U) & 0x1FU;
    uint32_t mantissa = value & 0x03FFU;
    uint32_t bits = 0;
    if (exponent == 0) {
        if (mantissa == 0) {
            bits = sign;
        } else {
            int32_t unbiasedExponent = -14;
            while ((mantissa & 0x0400U) == 0) {
                mantissa <<= 1U;
                --unbiasedExponent;
            }
            mantissa &= 0x03FFU;
            bits = sign | (static_cast<uint32_t>(unbiasedExponent + 127) << 23U) |
                (mantissa << 13U);
        }
    } else if (exponent == 31) {
        bits = sign | 0x7F800000U | (mantissa << 13U);
    } else {
        bits = sign | ((exponent + 112U) << 23U) | (mantissa << 13U);
    }
    float result = 0.0F;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

bool IsFloatType(OH_NN_DataType type)
{
    return type == OH_NN_FLOAT16 || type == OH_NN_FLOAT32;
}

size_t ElementSize(OH_NN_DataType type)
{
    if (type == OH_NN_FLOAT16) {
        return sizeof(uint16_t);
    }
    if (type == OH_NN_FLOAT32 || type == OH_NN_INT32) {
        return sizeof(uint32_t);
    }
    return 0;
}

void FillFloatTensor(void *destination, size_t count, OH_NN_DataType type, float value)
{
    if (type == OH_NN_FLOAT16) {
        std::fill_n(static_cast<uint16_t *>(destination), count, FloatToHalf(value));
    } else {
        std::fill_n(static_cast<float *>(destination), count, value);
    }
}

void SetFloatElement(void *destination, size_t index, OH_NN_DataType type, float value)
{
    if (type == OH_NN_FLOAT16) {
        static_cast<uint16_t *>(destination)[index] = FloatToHalf(value);
    } else {
        static_cast<float *>(destination)[index] = value;
    }
}

int32_t Argmax(const void *data, size_t count, OH_NN_DataType type)
{
    int32_t bestIndex = 0;
    float bestValue = -std::numeric_limits<float>::infinity();
    for (size_t index = 0; index < count; ++index) {
        const float value = type == OH_NN_FLOAT16
            ? HalfToFloat(static_cast<const uint16_t *>(data)[index])
            : static_cast<const float *>(data)[index];
        if (value > bestValue) {
            bestValue = value;
            bestIndex = static_cast<int32_t>(index);
        }
    }
    return bestIndex;
}

bool CopyExact(void *destination, size_t destinationSize, const void *source, size_t sourceSize,
               const char *stage, const std::string &label)
{
    if (destination == nullptr || source == nullptr || destinationSize != sourceSize) {
        DiagnosticLog::Fatal(stage, "SIZE_MISMATCH",
            label + " dst=" + std::to_string(destinationSize) + " src=" + std::to_string(sourceSize));
        return false;
    }
    std::memcpy(destination, source, sourceSize);
    return true;
}

bool ShapeEquals(const std::vector<int32_t> &actual, std::initializer_list<int32_t> expected)
{
    return actual == std::vector<int32_t>(expected);
}
}

UniRecEngine &UniRecEngine::Instance()
{
    static UniRecEngine instance;
    return instance;
}

bool UniRecEngine::BuildForGear(NativeResourceManager *manager, const Gear &gear)
{
    if (loadedGear_ == gear.name && encoder_.InputCount() == 1 && decoder_.InputCount() == 7) {
        DiagnosticLog::Info("MODEL_BUILD", "REUSE", gear.name);
        return true;
    }
    encoder_.Unload();
    decoder_.Unload();
    loadedGear_.clear();

    if (!tokenizer_.Ready() && !tokenizer_.Load(manager, "models/unirec_tokenizer.bin")) {
        return false;
    }
    if (!encoder_.Load(manager, "models/unirec_encoder.om", "encoder", EncoderShapes(gear))) {
        return false;
    }
    if (!decoder_.Load(manager, "models/unirec_decoder.om", "decoder", DecoderShapes(gear))) {
        return false;
    }
    if (!ValidateContract(gear)) {
        return false;
    }
    loadedGear_ = gear.name;
    return true;
}

bool UniRecEngine::ValidateContract(const Gear &gear)
{
    if (encoder_.InputCount() != 1 || encoder_.OutputCount() != 2 ||
        decoder_.InputCount() != 7 || decoder_.OutputCount() != 3) {
        DiagnosticLog::Fatal("IO_CONTRACT", "MODEL_COUNTS", "encoder=1/2 decoder=7/3 required");
        return false;
    }
    const auto expectedCross = std::vector<int32_t> {
        DECODER_LAYERS, DECODER_HEADS, gear.visualTokens, HEAD_DIM};
    if (encoder_.OutputShape(0) != expectedCross || encoder_.OutputShape(1) != expectedCross ||
        decoder_.InputShape(3) != expectedCross || decoder_.InputShape(4) != expectedCross) {
        DiagnosticLog::Fatal("IO_CONTRACT", "CROSS_KV_SHAPE", gear.name);
        return false;
    }
    if (!ShapeEquals(decoder_.OutputShape(0), {1, VOCAB_SIZE}) ||
        !ShapeEquals(decoder_.OutputShape(1), {DECODER_LAYERS, DECODER_HEADS, 1, HEAD_DIM}) ||
        !ShapeEquals(decoder_.OutputShape(2), {DECODER_LAYERS, DECODER_HEADS, 1, HEAD_DIM})) {
        DiagnosticLog::Fatal("IO_CONTRACT", "DECODER_OUTPUT_SHAPE", gear.name);
        return false;
    }
    if (encoder_.OutputType(0) != decoder_.InputType(3) ||
        encoder_.OutputType(1) != decoder_.InputType(4) ||
        decoder_.OutputType(1) != decoder_.InputType(5) ||
        decoder_.OutputType(2) != decoder_.InputType(6) ||
        !IsFloatType(encoder_.InputType(0)) || !IsFloatType(decoder_.OutputType(0)) ||
        decoder_.InputType(0) != OH_NN_INT32 || decoder_.InputType(1) != OH_NN_INT32) {
        DiagnosticLog::Fatal("IO_CONTRACT", "DTYPE_MISMATCH", gear.name);
        return false;
    }
    DiagnosticLog::Info("IO_CONTRACT", "VALIDATED", gear.name);
    return true;
}

bool UniRecEngine::PrepareImage(const std::vector<uint8_t> &bgraPixels, int32_t width, int32_t height,
                                const Gear &gear)
{
    if (width <= 0 || height <= 0 || width > gear.width || height > gear.height ||
        bgraPixels.size() != static_cast<size_t>(width) * height * 4U) {
        DiagnosticLog::Fatal("PREPROCESS", "PIXEL_SHAPE",
            "scaled=" + std::to_string(width) + "x" + std::to_string(height) +
            " bytes=" + std::to_string(bgraPixels.size()));
        return false;
    }
    void *destination = encoder_.InputData(0);
    const OH_NN_DataType type = encoder_.InputType(0);
    const size_t elementSize = ElementSize(type);
    const size_t plane = static_cast<size_t>(gear.width) * gear.height;
    if (destination == nullptr || elementSize == 0 || encoder_.InputSize(0) != plane * 3U * elementSize) {
        DiagnosticLog::Fatal("PREPROCESS", "INPUT_BUFFER", gear.name);
        return false;
    }
    FillFloatTensor(destination, plane * 3U, type, 1.0F);
    for (int32_t y = 0; y < height; ++y) {
        for (int32_t x = 0; x < width; ++x) {
            const size_t source = (static_cast<size_t>(y) * width + x) * 4U;
            const size_t target = static_cast<size_t>(y) * gear.width + x;
            const float blue = static_cast<float>(bgraPixels[source]) / 127.5F - 1.0F;
            const float green = static_cast<float>(bgraPixels[source + 1U]) / 127.5F - 1.0F;
            const float red = static_cast<float>(bgraPixels[source + 2U]) / 127.5F - 1.0F;
            SetFloatElement(destination, target, type, red);
            SetFloatElement(destination, plane + target, type, green);
            SetFloatElement(destination, plane * 2U + target, type, blue);
        }
    }
    DiagnosticLog::Info("PREPROCESS", "READY",
        "gear=" + std::string(gear.name) + " content=" + std::to_string(width) + "x" +
        std::to_string(height) + " padding=right,bottom");
    return true;
}

EngineResult UniRecEngine::Run(NativeResourceManager *manager,
                               const std::vector<uint8_t> &bgraPixels,
                               int32_t scaledWidth, int32_t scaledHeight,
                               const std::string &gearName, int32_t maxTokens)
{
    std::lock_guard<std::mutex> lock(mutex_);
    EngineResult result;
    result.gear = gearName;
    if (DiagnosticLog::IsStopped()) {
        result.errorCode = "LOG_TERMINATED";
        result.message = "A prior fatal error terminated the native pipeline; restart the app.";
        return result;
    }
    const Gear *gear = FindGear(gearName);
    if (manager == nullptr || gear == nullptr || maxTokens < 1 || maxTokens > CACHE_CAPACITY) {
        DiagnosticLog::Fatal("REQUEST", "INVALID_ARGUMENT", gearName);
        result.errorCode = "INVALID_ARGUMENT";
        result.message = "Invalid resource manager, gear, or maxTokens";
        return result;
    }

    const char *version = HMS_HiAI_GetVersion();
    DiagnosticLog::Info("BOOT", "CANN_VERSION", version == nullptr ? "unknown" : version);
    const auto totalBegin = Clock::now();
    if (!BuildForGear(manager, *gear) || !PrepareImage(bgraPixels, scaledWidth, scaledHeight, *gear)) {
        result.errorCode = "INITIALIZATION_FAILED";
        result.message = "See the final UniRecOM FATAL log";
        return result;
    }

    auto begin = Clock::now();
    DiagnosticLog::Info("ENCODER", "RUN_BEGIN", gearName);
    if (!encoder_.Run()) {
        result.errorCode = "ENCODER_FAILED";
        result.message = "See the final UniRecOM FATAL log";
        return result;
    }
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now() - begin).count();
    DiagnosticLog::Info("ENCODER", "RUN_END", "ms=" + std::to_string(elapsed));

    if (!CopyExact(decoder_.InputData(3), decoder_.InputSize(3), encoder_.OutputData(0),
                   encoder_.OutputSize(0), "ENCODER", "cross_k") ||
        !CopyExact(decoder_.InputData(4), decoder_.InputSize(4), encoder_.OutputData(1),
                   encoder_.OutputSize(1), "ENCODER", "cross_v")) {
        result.errorCode = "CROSS_KV_COPY_FAILED";
        result.message = "See the final UniRecOM FATAL log";
        return result;
    }

    std::memset(decoder_.InputData(5), 0, decoder_.InputSize(5));
    std::memset(decoder_.InputData(6), 0, decoder_.InputSize(6));
    const OH_NN_DataType maskType = decoder_.InputType(2);
    void *maskData = decoder_.InputData(2);
    FillFloatTensor(maskData, CACHE_CAPACITY + 1U, maskType, -10000.0F);
    SetFloatElement(maskData, CACHE_CAPACITY, maskType, 0.0F);
    DiagnosticLog::Info("DECODER", "CACHE_READY",
        "keyBytes=" + std::to_string(decoder_.InputSize(5)) +
        " valueBytes=" + std::to_string(decoder_.InputSize(6)));

    std::vector<int32_t> generated {BOS_TOKEN_ID};
    begin = Clock::now();
    for (int32_t step = 0; step < maxTokens - 1; ++step) {
        *static_cast<int32_t *>(decoder_.InputData(0)) = generated.back();
        *static_cast<int32_t *>(decoder_.InputData(1)) = PAD_TOKEN_ID + 1 + step;
        if (!decoder_.Run()) {
            result.errorCode = "DECODER_FAILED";
            result.message = "See the final UniRecOM FATAL log";
            return result;
        }
        const int32_t nextToken = Argmax(
            decoder_.OutputData(0), VOCAB_SIZE, decoder_.OutputType(0));
        generated.push_back(nextToken);

        const size_t cacheElementSize = ElementSize(decoder_.InputType(5));
        const size_t headBlock = static_cast<size_t>(HEAD_DIM) * cacheElementSize;
        const auto *newKeys = static_cast<const uint8_t *>(decoder_.OutputData(1));
        const auto *newValues = static_cast<const uint8_t *>(decoder_.OutputData(2));
        auto *pastKeys = static_cast<uint8_t *>(decoder_.InputData(5));
        auto *pastValues = static_cast<uint8_t *>(decoder_.InputData(6));
        for (int32_t layer = 0; layer < DECODER_LAYERS; ++layer) {
            for (int32_t head = 0; head < DECODER_HEADS; ++head) {
                const size_t newIndex = static_cast<size_t>(layer * DECODER_HEADS + head) * headBlock;
                const size_t cacheIndex =
                    (static_cast<size_t>(layer * DECODER_HEADS + head) * CACHE_CAPACITY + step) * headBlock;
                std::memcpy(pastKeys + cacheIndex, newKeys + newIndex, headBlock);
                std::memcpy(pastValues + cacheIndex, newValues + newIndex, headBlock);
            }
        }
        SetFloatElement(maskData, static_cast<size_t>(step), maskType, 0.0F);

        if (step < 4 || (step + 1) % 25 == 0 || nextToken == EOS_TOKEN_ID) {
            DiagnosticLog::Info("DECODER", "STEP",
                "step=" + std::to_string(step) + " token=" + std::to_string(nextToken));
        }
        if (nextToken == EOS_TOKEN_ID) {
            break;
        }
    }
    elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now() - begin).count();
    result.text = tokenizer_.Decode(generated);
    result.ok = true;
    DiagnosticLog::Info("POSTPROCESS", "DECODED",
        "tokens=" + std::to_string(generated.size()) + " chars=" + std::to_string(result.text.size()) +
        " decoderMs=" + std::to_string(elapsed));
    const auto totalMs = std::chrono::duration_cast<std::chrono::milliseconds>(
        Clock::now() - totalBegin).count();
    DiagnosticLog::Info("DONE", "SUCCESS", "gear=" + gearName + " totalMs=" + std::to_string(totalMs));
    return result;
}

void UniRecEngine::Unload()
{
    std::lock_guard<std::mutex> lock(mutex_);
    encoder_.Unload();
    decoder_.Unload();
    loadedGear_.clear();
}
