#ifndef UNIREC_ENGINE_H
#define UNIREC_ENGINE_H

#include "om_model.h"
#include "tokenizer.h"

#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

struct NativeResourceManager;

struct EngineResult {
    bool ok {false};
    std::string text;
    std::string gear;
    std::string errorCode;
    std::string message;
};

class UniRecEngine {
public:
    static UniRecEngine &Instance();
    EngineResult Run(NativeResourceManager *manager, const std::vector<uint8_t> &bgraPixels,
                     int32_t scaledWidth, int32_t scaledHeight, const std::string &gearName,
                     int32_t maxTokens);
    void Unload();

private:
    bool BuildForGear(NativeResourceManager *manager, const struct Gear &gear);
    bool PrepareImage(const std::vector<uint8_t> &bgraPixels, int32_t width, int32_t height,
                      const struct Gear &gear);
    bool ValidateContract(const struct Gear &gear);

    std::mutex mutex_;
    OmModel encoder_;
    OmModel decoder_;
    Tokenizer tokenizer_;
    std::string loadedGear_;
};

#endif

