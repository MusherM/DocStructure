#include "diagnostic_log.h"
#include "unirec_engine.h"

#include <memory>
#include <napi/native_api.h>
#include <rawfile/raw_file_manager.h>
#include <string>
#include <vector>

namespace {
struct AsyncContext {
    napi_env env {nullptr};
    napi_async_work work {nullptr};
    napi_deferred deferred {nullptr};
    NativeResourceManager *resourceManager {nullptr};
    std::vector<uint8_t> pixels;
    int32_t width {0};
    int32_t height {0};
    int32_t maxTokens {0};
    std::string gear;
    EngineResult result;
};

std::string GetString(napi_env env, napi_value value)
{
    size_t length = 0;
    napi_get_value_string_utf8(env, value, nullptr, 0, &length);
    std::string result(length, '\0');
    napi_get_value_string_utf8(env, value, result.data(), length + 1, &length);
    return result;
}

napi_value MakeString(napi_env env, const std::string &value)
{
    napi_value result = nullptr;
    napi_create_string_utf8(env, value.c_str(), value.size(), &result);
    return result;
}

void SetString(napi_env env, napi_value object, const char *name, const std::string &value)
{
    napi_set_named_property(env, object, name, MakeString(env, value));
}

void Execute(napi_env, void *data)
{
    auto *context = static_cast<AsyncContext *>(data);
    context->result = UniRecEngine::Instance().Run(
        context->resourceManager, context->pixels, context->width, context->height,
        context->gear, context->maxTokens);
}

void Complete(napi_env env, napi_status status, void *data)
{
    std::unique_ptr<AsyncContext> context(static_cast<AsyncContext *>(data));
    if (status != napi_ok && context->result.errorCode.empty()) {
        context->result.ok = false;
        context->result.errorCode = "ASYNC_WORK_FAILED";
        context->result.message = "Native async work did not complete";
    }
    napi_value object = nullptr;
    napi_value ok = nullptr;
    napi_create_object(env, &object);
    napi_get_boolean(env, context->result.ok, &ok);
    napi_set_named_property(env, object, "ok", ok);
    SetString(env, object, "text", context->result.text);
    SetString(env, object, "gear", context->result.gear);
    SetString(env, object, "errorCode", context->result.errorCode);
    SetString(env, object, "message", context->result.message);
    napi_resolve_deferred(env, context->deferred, object);
    if (context->resourceManager != nullptr) {
        OH_ResourceManager_ReleaseNativeResourceManager(context->resourceManager);
    }
    napi_delete_async_work(env, context->work);
}

napi_value RunInference(napi_env env, napi_callback_info info)
{
    size_t argc = 6;
    napi_value argv[6] = {nullptr, nullptr, nullptr, nullptr, nullptr, nullptr};
    napi_get_cb_info(env, info, &argc, argv, nullptr, nullptr);
    if (argc != 6) {
        napi_throw_type_error(env, nullptr, "runInference expects 6 arguments");
        return nullptr;
    }

    napi_typedarray_type arrayType;
    size_t length = 0;
    void *pixelData = nullptr;
    napi_value arrayBuffer = nullptr;
    size_t byteOffset = 0;
    if (napi_get_typedarray_info(env, argv[1], &arrayType, &length, &pixelData,
        &arrayBuffer, &byteOffset) != napi_ok || arrayType != napi_uint8_array || pixelData == nullptr) {
        napi_throw_type_error(env, nullptr, "bgraPixels must be Uint8Array");
        return nullptr;
    }

    auto context = std::make_unique<AsyncContext>();
    context->env = env;
    context->resourceManager = OH_ResourceManager_InitNativeResourceManager(env, argv[0]);
    context->pixels.assign(static_cast<uint8_t *>(pixelData), static_cast<uint8_t *>(pixelData) + length);
    napi_get_value_int32(env, argv[2], &context->width);
    napi_get_value_int32(env, argv[3], &context->height);
    context->gear = GetString(env, argv[4]);
    napi_get_value_int32(env, argv[5], &context->maxTokens);

    napi_value promise = nullptr;
    napi_create_promise(env, &context->deferred, &promise);
    napi_value resourceName = MakeString(env, "UniRecDynamicOmInference");
    napi_create_async_work(env, nullptr, resourceName, Execute, Complete, context.get(), &context->work);
    napi_queue_async_work(env, context->work);
    context.release();
    return promise;
}

napi_value Unload(napi_env env, napi_callback_info)
{
    UniRecEngine::Instance().Unload();
    napi_value result = nullptr;
    napi_get_undefined(env, &result);
    return result;
}
}

EXTERN_C_START
static napi_value Init(napi_env env, napi_value exports)
{
    napi_property_descriptor properties[] = {
        {"runInference", nullptr, RunInference, nullptr, nullptr, nullptr, napi_default, nullptr},
        {"unload", nullptr, Unload, nullptr, nullptr, nullptr, napi_default, nullptr},
    };
    napi_define_properties(env, exports, sizeof(properties) / sizeof(properties[0]), properties);
    return exports;
}
EXTERN_C_END

static napi_module module = {
    .nm_version = 1,
    .nm_flags = 0,
    .nm_filename = nullptr,
    .nm_register_func = Init,
    .nm_modname = "entry",
    .nm_priv = nullptr,
    .reserved = {0},
};

extern "C" __attribute__((constructor)) void RegisterModule()
{
    napi_module_register(&module);
}
