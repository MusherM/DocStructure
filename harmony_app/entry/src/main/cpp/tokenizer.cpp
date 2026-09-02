#include "tokenizer.h"

#include "diagnostic_log.h"
#include "model_contract.h"

#include <cstring>
#include <rawfile/raw_file_manager.h>
#include <sstream>
#include <vector>

namespace {
constexpr uint32_t TOKENIZER_VERSION = 1;

uint32_t ReadU32(const std::vector<uint8_t> &data, size_t &offset, bool &ok)
{
    if (offset + sizeof(uint32_t) > data.size()) {
        ok = false;
        return 0;
    }
    uint32_t value = 0;
    std::memcpy(&value, data.data() + offset, sizeof(value));
    offset += sizeof(value);
    return value;
}

void ReplaceAll(std::string &text, const std::string &source, const std::string &target)
{
    size_t position = 0;
    while ((position = text.find(source, position)) != std::string::npos) {
        text.replace(position, source.size(), target);
        position += target.size();
    }
}
}

bool Tokenizer::Load(NativeResourceManager *manager, const char *path)
{
    RawFile *file = OH_ResourceManager_OpenRawFile(manager, path);
    if (file == nullptr) {
        DiagnosticLog::Fatal("TOKENIZER", "ASSET_OPEN", path);
        return false;
    }
    const long length = OH_ResourceManager_GetRawFileSize(file);
    if (length <= 24) {
        OH_ResourceManager_CloseRawFile(file);
        DiagnosticLog::Fatal("TOKENIZER", "ASSET_SIZE", std::to_string(length));
        return false;
    }
    std::vector<uint8_t> data(static_cast<size_t>(length));
    const int read = OH_ResourceManager_ReadRawFile(file, data.data(), data.size());
    OH_ResourceManager_CloseRawFile(file);
    if (read != length) {
        DiagnosticLog::Fatal("TOKENIZER", "ASSET_READ", std::to_string(read));
        return false;
    }

    if (std::memcmp(data.data(), "URTK", 4) != 0) {
        DiagnosticLog::Fatal("TOKENIZER", "BAD_MAGIC", path);
        return false;
    }
    size_t offset = 4;
    bool ok = true;
    const uint32_t version = ReadU32(data, offset, ok);
    const uint32_t vocabularySize = ReadU32(data, offset, ok);
    bos_ = ReadU32(data, offset, ok);
    eos_ = ReadU32(data, offset, ok);
    pad_ = ReadU32(data, offset, ok);
    if (!ok || version != TOKENIZER_VERSION || vocabularySize != VOCAB_SIZE ||
        bos_ != BOS_TOKEN_ID || eos_ != EOS_TOKEN_ID || pad_ != PAD_TOKEN_ID) {
        DiagnosticLog::Fatal("TOKENIZER", "BAD_HEADER", "version/vocab/special-token mismatch");
        return false;
    }

    tokens_.clear();
    tokens_.reserve(vocabularySize);
    for (uint32_t index = 0; index < vocabularySize; ++index) {
        const uint32_t tokenLength = ReadU32(data, offset, ok);
        if (!ok || offset + tokenLength > data.size()) {
            DiagnosticLog::Fatal("TOKENIZER", "TRUNCATED", std::to_string(index));
            tokens_.clear();
            return false;
        }
        tokens_.emplace_back(reinterpret_cast<const char *>(data.data() + offset), tokenLength);
        offset += tokenLength;
    }
    DiagnosticLog::Info("TOKENIZER", "READY", "vocab=56371 bytes=" + std::to_string(length));
    return true;
}

std::string Tokenizer::Decode(const std::vector<int32_t> &tokenIds) const
{
    std::string text;
    for (const int32_t tokenId : tokenIds) {
        if (tokenId >= 0 && static_cast<size_t>(tokenId) < tokens_.size()) {
            text += tokens_[static_cast<size_t>(tokenId)];
        }
    }
    ReplaceAll(text, "Ġ", " ");
    ReplaceAll(text, "Ċ", "\n");
    ReplaceAll(text, "<|bos|>", "");
    ReplaceAll(text, "<|eos|>", "");
    ReplaceAll(text, "<|pad|>", "");
    ReplaceAll(text, "<|unk|>", "");
    ReplaceAll(text, "<|sn|>", " ");
    ReplaceAll(text, "<s>", "");
    ReplaceAll(text, "</s>", "");
    ReplaceAll(text, "\xEF\xBF\xBF", "");
    const size_t first = text.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) {
        return "";
    }
    const size_t last = text.find_last_not_of(" \t\r\n");
    return text.substr(first, last - first + 1);
}

bool Tokenizer::Ready() const
{
    return tokens_.size() == VOCAB_SIZE;
}
