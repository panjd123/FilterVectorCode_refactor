#pragma once

// Full Knowhere obtains this interface from a Conan package. The flat-HNSW
// adapter only encounters it as a shared_ptr template argument.
namespace milvus { class FileManager; }
