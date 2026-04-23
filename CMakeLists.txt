cmake_minimum_required(VERSION 3.28)

project(GEMM_VULKAN LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(Vulkan REQUIRED)

find_program(GEMM_SHADER_COMPILER NAMES glslc glslangValidator REQUIRED)

set(GEMM_SHADER_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/shaders)
set(GEMM_SHADER_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/shaders)

set(GEMM_SHADER_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/shaders/sgemm/00_naive.comp
    ${CMAKE_CURRENT_SOURCE_DIR}/shaders/sgemm/01_coalesced.comp
    ${CMAKE_CURRENT_SOURCE_DIR}/shaders/sgemm/02_shared_tiling.comp
    ${CMAKE_CURRENT_SOURCE_DIR}/shaders/sgemm/03_thread_tiling_1d.comp
    ${CMAKE_CURRENT_SOURCE_DIR}/shaders/sgemm/03_thread_tiling_1d_shared_vec.comp
    ${CMAKE_CURRENT_SOURCE_DIR}/shaders/sgemm/04_thread_tiling_2d.comp
    ${CMAKE_CURRENT_SOURCE_DIR}/shaders/sgemm/05_vectorized.comp
    ${CMAKE_CURRENT_SOURCE_DIR}/shaders/sgemm/06_bank_conflict_avoid.comp
    ${CMAKE_CURRENT_SOURCE_DIR}/shaders/sgemm/07_bank_conflict_padding.comp
    ${CMAKE_CURRENT_SOURCE_DIR}/shaders/sgemm/08_subgroup_tiling.comp
    ${CMAKE_CURRENT_SOURCE_DIR}/shaders/sgemm/08_subgroup_tiling_compiler_friendly.comp
    ${CMAKE_CURRENT_SOURCE_DIR}/shaders/sgemm/09_double_buffering.comp
    ${CMAKE_CURRENT_SOURCE_DIR}/shaders/sgemm/10_tile_swizzle.comp
    ${CMAKE_CURRENT_SOURCE_DIR}/shaders/sgemm/11_autotuned.comp
    ${CMAKE_CURRENT_SOURCE_DIR}/shaders/sgemm/12_split_k_partial.comp
    ${CMAKE_CURRENT_SOURCE_DIR}/shaders/sgemm/12_split_k_reduce.comp
    ${CMAKE_CURRENT_SOURCE_DIR}/shaders/sgemm/13_persistent_scheduler.comp
    ${CMAKE_CURRENT_SOURCE_DIR}/shaders/sgemm/14_stream_k_partial.comp
)

function(compile_gemm_shader INPUT_FILE OUTPUT_FILE)
  get_filename_component(OUTPUT_DIR "${OUTPUT_FILE}" DIRECTORY)
  get_filename_component(INPUT_NAME "${INPUT_FILE}" NAME)
  if(GEMM_SHADER_COMPILER MATCHES "glslc$")
    set(SHADER_ARGS
        -I${GEMM_SHADER_SOURCE_DIR}
        -o ${OUTPUT_FILE}
        ${INPUT_FILE}
    )
    if(INPUT_NAME STREQUAL "08_subgroup_tiling.comp")
      list(PREPEND SHADER_ARGS -O)
    endif()
  else()
    set(SHADER_ARGS
        -V
        -I${GEMM_SHADER_SOURCE_DIR}
        -o ${OUTPUT_FILE}
        ${INPUT_FILE}
    )
  endif()

  add_custom_command(
      OUTPUT ${OUTPUT_FILE}
      COMMAND ${CMAKE_COMMAND} -E make_directory ${OUTPUT_DIR}
      COMMAND ${GEMM_SHADER_COMPILER} ${SHADER_ARGS}
      DEPENDS
          ${INPUT_FILE}
          ${CMAKE_CURRENT_SOURCE_DIR}/shaders/common/common.glsl
          ${CMAKE_CURRENT_SOURCE_DIR}/shaders/common/bounds.glsl
          ${CMAKE_CURRENT_SOURCE_DIR}/shaders/common/loads.glsl
      VERBATIM
  )
endfunction()

set(GEMM_SHADER_OUTPUTS)
foreach(SHADER_SOURCE ${GEMM_SHADER_SOURCES})
  file(RELATIVE_PATH RELATIVE_SHADER ${GEMM_SHADER_SOURCE_DIR} ${SHADER_SOURCE})
  set(SHADER_OUTPUT ${GEMM_SHADER_BINARY_DIR}/${RELATIVE_SHADER}.spv)
  compile_gemm_shader(${SHADER_SOURCE} ${SHADER_OUTPUT})
  list(APPEND GEMM_SHADER_OUTPUTS ${SHADER_OUTPUT})
endforeach()

add_custom_target(gemm_vulkan_shaders ALL DEPENDS ${GEMM_SHADER_OUTPUTS})

set(GEMM_VULKAN_SOURCES
    src/app/main.cpp
    src/runtime/instance.cpp
    src/runtime/device.cpp
    src/runtime/buffer.cpp
    src/runtime/descriptor.cpp
    src/runtime/pipeline.cpp
    src/runtime/command.cpp
    src/runtime/timestamp.cpp
    src/gemm/reference_cpu.cpp
    src/gemm/verifier.cpp
    src/gemm/benchmark.cpp
    src/gemm/runner.cpp
    src/gemm/kernels/registry.cpp
    src/gemm/kernels/launch_config.cpp
    src/gemm/configs/default_configs.cpp
    src/gemm/configs/autotune_space.cpp
)

add_executable(gemm_vulkan ${GEMM_VULKAN_SOURCES})
add_dependencies(gemm_vulkan gemm_vulkan_shaders)

target_include_directories(
    gemm_vulkan
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_compile_definitions(
    gemm_vulkan
    PRIVATE GEMM_VULKAN_SHADER_DIR="${GEMM_SHADER_BINARY_DIR}"
)

target_link_libraries(gemm_vulkan PRIVATE Vulkan::Vulkan)
