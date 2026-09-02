#ifndef UNIREC_MODEL_CONTRACT_H
#define UNIREC_MODEL_CONTRACT_H

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

struct Gear {
    const char *name;
    int32_t width;
    int32_t height;
    int32_t visualTokens;
};

inline constexpr std::array<Gear, 4> GEARS {{
    {"384x512", 384, 512, 192},
    {"576x768", 576, 768, 432},
    {"768x1024", 768, 1024, 768},
    {"960x1408", 960, 1408, 1320},
}};

inline constexpr int32_t DECODER_LAYERS = 6;
inline constexpr int32_t DECODER_HEADS = 6;
inline constexpr int32_t HEAD_DIM = 128;
inline constexpr int32_t CACHE_CAPACITY = 2048;
inline constexpr int32_t VOCAB_SIZE = 56371;
inline constexpr int32_t BOS_TOKEN_ID = 0;
inline constexpr int32_t PAD_TOKEN_ID = 1;
inline constexpr int32_t EOS_TOKEN_ID = 2;

inline const Gear *FindGear(const std::string &name)
{
    for (const auto &gear : GEARS) {
        if (name == gear.name) {
            return &gear;
        }
    }
    return nullptr;
}

inline std::vector<std::vector<int32_t>> EncoderShapes(const Gear &gear)
{
    return {{1, 3, gear.height, gear.width}};
}

inline std::vector<std::vector<int32_t>> DecoderShapes(const Gear &gear)
{
    return {
        {1, 1},
        {1, 1},
        {1, 1, 1, CACHE_CAPACITY + 1},
        {DECODER_LAYERS, DECODER_HEADS, gear.visualTokens, HEAD_DIM},
        {DECODER_LAYERS, DECODER_HEADS, gear.visualTokens, HEAD_DIM},
        {DECODER_LAYERS, DECODER_HEADS, CACHE_CAPACITY, HEAD_DIM},
        {DECODER_LAYERS, DECODER_HEADS, CACHE_CAPACITY, HEAD_DIM},
    };
}

#endif
