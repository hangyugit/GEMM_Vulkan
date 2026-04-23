#include <array>
#include <fstream>
#include <utility>

#include "runtime/runtime.hpp"

namespace gemm::runtime {

ComputePipeline::~ComputePipeline() {
  if (pipeline != VK_NULL_HANDLE && device != VK_NULL_HANDLE) {
    vkDestroyPipeline(device, pipeline, nullptr);
  }
  if (pipeline_layout != VK_NULL_HANDLE && device != VK_NULL_HANDLE) {
    vkDestroyPipelineLayout(device, pipeline_layout, nullptr);
  }
  if (shader_module != VK_NULL_HANDLE && device != VK_NULL_HANDLE) {
    vkDestroyShaderModule(device, shader_module, nullptr);
  }
}

ComputePipeline::ComputePipeline(ComputePipeline&& other) noexcept
    : device(std::exchange(other.device, VK_NULL_HANDLE)),
      shader_module(std::exchange(other.shader_module, VK_NULL_HANDLE)),
      pipeline_layout(std::exchange(other.pipeline_layout, VK_NULL_HANDLE)),
      pipeline(std::exchange(other.pipeline, VK_NULL_HANDLE)) {}

ComputePipeline& ComputePipeline::operator=(ComputePipeline&& other) noexcept {
  if (this != &other) {
    if (pipeline != VK_NULL_HANDLE && device != VK_NULL_HANDLE) {
      vkDestroyPipeline(device, pipeline, nullptr);
    }
    if (pipeline_layout != VK_NULL_HANDLE && device != VK_NULL_HANDLE) {
      vkDestroyPipelineLayout(device, pipeline_layout, nullptr);
    }
    if (shader_module != VK_NULL_HANDLE && device != VK_NULL_HANDLE) {
      vkDestroyShaderModule(device, shader_module, nullptr);
    }
    device = std::exchange(other.device, VK_NULL_HANDLE);
    shader_module = std::exchange(other.shader_module, VK_NULL_HANDLE);
    pipeline_layout = std::exchange(other.pipeline_layout, VK_NULL_HANDLE);
    pipeline = std::exchange(other.pipeline, VK_NULL_HANDLE);
  }
  return *this;
}

std::vector<std::uint32_t> read_spirv_file(const std::filesystem::path& path) {
  std::ifstream stream(path, std::ios::binary | std::ios::ate);
  if (!stream) {
    throw std::runtime_error("Unable to open shader file: " + path.string());
  }

  const std::streamsize size = stream.tellg();
  if (size <= 0 || (size % 4) != 0) {
    throw std::runtime_error("Invalid SPIR-V file size: " + path.string());
  }

  stream.seekg(0, std::ios::beg);
  std::vector<std::uint32_t> words(static_cast<std::size_t>(size) / 4U);
  stream.read(reinterpret_cast<char*>(words.data()), size);
  if (!stream) {
    throw std::runtime_error("Failed to read SPIR-V file: " + path.string());
  }
  return words;
}

ComputePipeline create_compute_pipeline(const DeviceContext& device,
                                        const DescriptorResources& descriptors,
                                        const std::filesystem::path& spirv_path,
                                        std::uint32_t push_constant_size,
                                        const AutotuneSpecialization* specialization) {
  ComputePipeline pipeline{};
  pipeline.device = device.device;

  const std::vector<std::uint32_t> code = read_spirv_file(spirv_path);

  VkShaderModuleCreateInfo shader_info{};
  shader_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
  shader_info.codeSize = code.size() * sizeof(std::uint32_t);
  shader_info.pCode = code.data();
  check_vk(vkCreateShaderModule(device.device, &shader_info, nullptr,
                                &pipeline.shader_module),
           "vkCreateShaderModule");

  VkPushConstantRange push_constant_range{};
  push_constant_range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
  push_constant_range.offset = 0;
  push_constant_range.size = push_constant_size;

  VkPipelineLayoutCreateInfo layout_info{};
  layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
  layout_info.setLayoutCount = 1;
  layout_info.pSetLayouts = &descriptors.layout;
  if (push_constant_size > 0) {
    layout_info.pushConstantRangeCount = 1;
    layout_info.pPushConstantRanges = &push_constant_range;
  }
  check_vk(vkCreatePipelineLayout(device.device, &layout_info, nullptr,
                                  &pipeline.pipeline_layout),
           "vkCreatePipelineLayout");

  std::array<VkSpecializationMapEntry, 10> specialization_entries = {{
      {0, offsetof(AutotuneSpecialization, local_size_x), sizeof(std::uint32_t)},
      {1, offsetof(AutotuneSpecialization, block_m), sizeof(std::uint32_t)},
      {2, offsetof(AutotuneSpecialization, block_n), sizeof(std::uint32_t)},
      {3, offsetof(AutotuneSpecialization, block_k), sizeof(std::uint32_t)},
      {4, offsetof(AutotuneSpecialization, warp_m), sizeof(std::uint32_t)},
      {5, offsetof(AutotuneSpecialization, warp_n), sizeof(std::uint32_t)},
      {6, offsetof(AutotuneSpecialization, warp_n_iter), sizeof(std::uint32_t)},
      {7, offsetof(AutotuneSpecialization, thread_m), sizeof(std::uint32_t)},
      {8, offsetof(AutotuneSpecialization, thread_n), sizeof(std::uint32_t)},
      {9, offsetof(AutotuneSpecialization, stages), sizeof(std::uint32_t)},
  }};

  VkSpecializationInfo specialization_info{};
  specialization_info.mapEntryCount =
      static_cast<std::uint32_t>(specialization_entries.size());
  specialization_info.pMapEntries = specialization_entries.data();
  specialization_info.dataSize = sizeof(AutotuneSpecialization);
  specialization_info.pData = specialization;

  VkPipelineShaderStageCreateInfo stage_info{};
  stage_info.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  stage_info.stage = VK_SHADER_STAGE_COMPUTE_BIT;
  stage_info.module = pipeline.shader_module;
  stage_info.pSpecializationInfo =
      specialization != nullptr ? &specialization_info : nullptr;
  stage_info.pName = "main";

  VkComputePipelineCreateInfo pipeline_info{};
  pipeline_info.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
  pipeline_info.stage = stage_info;
  pipeline_info.layout = pipeline.pipeline_layout;

  check_vk(
      vkCreateComputePipelines(device.device, VK_NULL_HANDLE, 1, &pipeline_info,
                               nullptr, &pipeline.pipeline),
      "vkCreateComputePipelines");
  return pipeline;
}

}  // namespace gemm::runtime
