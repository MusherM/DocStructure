#ifndef UNIREC_DIAGNOSTIC_LOG_H
#define UNIREC_DIAGNOSTIC_LOG_H

#include <atomic>
#include <cstdint>
#include <string>

class DiagnosticLog {
public:
    static void Info(const char *stage, const char *event, const std::string &detail);
    static void Fatal(const char *stage, const char *code, const std::string &detail);
    static bool IsStopped();

private:
    static std::atomic<bool> stopped_;
    static std::atomic<uint64_t> sequence_;
};

#endif

