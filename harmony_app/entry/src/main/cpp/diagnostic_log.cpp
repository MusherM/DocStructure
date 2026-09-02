#include "diagnostic_log.h"

#include <hilog/log.h>

#undef LOG_DOMAIN
#define LOG_DOMAIN 0xD010
#undef LOG_TAG
#define LOG_TAG "UniRecOM"

std::atomic<bool> DiagnosticLog::stopped_ {false};
std::atomic<uint64_t> DiagnosticLog::sequence_ {0};

void DiagnosticLog::Info(const char *stage, const char *event, const std::string &detail)
{
    if (stopped_.load(std::memory_order_acquire)) {
        return;
    }
    const auto sequence = sequence_.fetch_add(1, std::memory_order_relaxed) + 1;
    OH_LOG_INFO(LOG_APP,
        "{\"seq\":%{public}llu,\"level\":\"INFO\",\"stage\":\"%{public}s\","
        "\"event\":\"%{public}s\",\"detail\":\"%{public}s\"}",
        static_cast<unsigned long long>(sequence), stage, event, detail.c_str());
}

void DiagnosticLog::Fatal(const char *stage, const char *code, const std::string &detail)
{
    bool expected = false;
    if (!stopped_.compare_exchange_strong(expected, true, std::memory_order_acq_rel)) {
        return;
    }
    const auto sequence = sequence_.fetch_add(1, std::memory_order_relaxed) + 1;
    OH_LOG_FATAL(LOG_APP,
        "{\"seq\":%{public}llu,\"level\":\"FATAL\",\"stage\":\"%{public}s\","
        "\"code\":\"%{public}s\",\"detail\":\"%{public}s\",\"terminal\":true}",
        static_cast<unsigned long long>(sequence), stage, code, detail.c_str());
}

bool DiagnosticLog::IsStopped()
{
    return stopped_.load(std::memory_order_acquire);
}

