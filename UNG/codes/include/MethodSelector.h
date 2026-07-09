#pragma once

#include <string>
#include <vector>
#include <memory>

#include "../third_party/onnxruntime-linux-x64-1.16.3/include/onnxruntime_cxx_api.h"

class MethodSelector
{
public:
    explicit MethodSelector(const std::string &model_path);

    // Returns true when the ONNX selector chooses the alternate route encoded
    // by the trained model.
    bool predict(const std::vector<float> &features);

private:
    Ort::Env _env;
    Ort::Session _session;
    Ort::AllocatorWithDefaultOptions _allocator;

    // Cached model metadata. The name pointers reference the stable strings
    // below and are reused for each inference call.
    std::vector<const char *> _input_node_names;
    std::vector<const char *> _output_node_names;
    std::vector<int64_t> _input_node_dims;

    std::string _input_name_str;
    std::string _output_name_str;
};
