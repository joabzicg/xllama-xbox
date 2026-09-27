// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT
#include "xllama/model_provision.h"
#include "xllama/path_utils.h"
#include <cassert>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>

int main() {
    namespace fs = std::filesystem;
    const auto dir = fs::temp_directory_path() /
                     ("xllama-projector-test-" +
                      std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    fs::create_directories(dir);
    struct Cleanup {
        fs::path path;
        ~Cleanup() {
            fs::remove_all(path);
        }
    } cleanup{dir};
    const auto projector = dir / "mmproj-Qwen2.5-VL-Q8_0.gguf";
    std::ofstream(projector) << "stub projector";
    assert(xllama::first_gguf_in_dir(dir.string()).empty());
    assert(!xllama::model_dir_files_ready({"mmproj-Qwen2.5-VL-Q8_0.gguf"}));
    assert(xllama::model_dir_files_ready({"mmproj-Qwen2.5-VL-Q8_0.gguf", "model.gguf"}));
    assert(xllama::find_mmproj_in_dir(dir.string()) == projector.u8string());
    const auto model = dir / "Qwen2.5-VL-Q4_K_M.gguf";
    std::ofstream(model) << "stub model";
    assert(xllama::first_gguf_in_dir(dir.string()) == model.string());
    assert(xllama::find_mmproj_in_dir(model.string()) == projector.u8string());
    std::ofstream(dir / "mmproj-other.gguf") << "ambiguous projector";
    assert(xllama::find_mmproj_in_dir(dir.string()).empty());
    assert(xllama::first_gguf_in_dir(dir.string()) == model.string());
    assert(xllama::find_mmproj_in_dir((dir / "missing" / "model.gguf").string()).empty());
    std::cout << "model/projector selection: PASS\n";
}
