#ifndef UNIREC_TOKENIZER_H
#define UNIREC_TOKENIZER_H

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

struct NativeResourceManager;

class Tokenizer {
public:
    bool Load(NativeResourceManager *manager, const char *path);
    std::string Decode(const std::vector<int32_t> &tokenIds) const;
    bool Ready() const;

private:
    std::vector<std::string> tokens_;
    uint32_t bos_ {0};
    uint32_t eos_ {2};
    uint32_t pad_ {1};
};

#endif

