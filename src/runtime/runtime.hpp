#pragma once

#include <vulkan/vulkan.h>

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string_view>
#include <vector>

namespace gemm::runtime {

void check_vk(VkResult result, std::string_view message);

struct Instance {
  VkInstance handle = VK_NULL_HANDLE;

  Instance() = default;
  ~Instance();
  Instance(const Instance&) = delete;
  Instance& operator=(const Instance&) = delete;
  Instance(Instance&& other) noexcept;
  Instance& operator=(Instance&& other) noexcept;
};

struct DeviceContext {
  VkPhysicalDevice physical_device = VK_NULL_HANDLE;
  VkDevice device = VK_NULL_HANDLE;
  VkQueue compute_queue = VK_NULL_HANDLE;
  std::uint32_t queue_family_index = 0;
  float timestamp_period = 0.0F;
  bool supports_timestamps = false;

  DeviceContext() = default;
  ~DeviceContext();
  DeviceContext(const DeviceContext&) = delete;
  DeviceContext& operator=(const DeviceContext&) = delete;
  DeviceContext(DeviceContext&& other) noexcept;
  DeviceContext& operator=(DeviceContext&& other) noexcept;
};

struct Buffer {
  VkDevice device = VK_NULL_HANDLE;
  VkBuffer buffer = VK_NULL_HANDLE;
  VkDeviceMemory memory = VK_NULL_HANDLE;
  void* mapped = nullptr;
  VkDeviceSize size = 0;

  Buffer() = default;
  ~Buffer();
  Buffer(const Buffer&) = delete;
  Buffer& operator=(const Buffer&) = delete;
  Buffer(Buffer&& other) noexcept;
  Buffer& operator=(Buffer&& other) noexcept;
};

struct DescriptorResources {
  VkDevice device = VK_NULL_HANDLE;
  VkDescriptorSetLayout layout = VK_NULL_HANDLE;
  VkDescriptorPool pool = VK_NULL_HANDLE;
  VkDescriptorSet set = VK_NULL_HANDLE;

  DescriptorResources() = default;
  ~DescriptorResources();
  DescriptorResources(const DescriptorResources&) = delete;
  DescriptorResources& operator=(const DescriptorResources&) = delete;
  DescriptorResources(DescriptorResources&& other) noexcept;
  DescriptorResources& operator=(DescriptorResources&& other) noexcept;
};

struct ComputePipeline {
  VkDevice device = VK_NULL_HANDLE;
  VkShaderModule shader_module = VK_NULL_HANDLE;
  VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
  VkPipeline pipeline = VK_NULL_HANDLE;

  ComputePipeline() = default;
  ~ComputePipeline();
  ComputePipeline(const ComputePipeline&) = delete;
  ComputePipeline& operator=(const ComputePipeline&) = delete;
  ComputePipeline(ComputePipeline&& other) noexcept;
  ComputePipeline& operator=(ComputePipeline&& other) noexcept;
};

struct CommandResources {
  VkDevice device = VK_NULL_HANDLE;
  VkCommandPool command_pool = VK_NULL_HANDLE;
  VkCommandBuffer command_buffer = VK_NULL_HANDLE;

  CommandResources() = default;
  ~CommandResources();
  CommandResources(const CommandResources&) = delete;
  CommandResources& operator=(const CommandResources&) = delete;
  CommandResources(CommandResources&& other) noexcept;
  CommandResources& operator=(CommandResources&& other) noexcept;
};

struct TimestampQuery {
  VkDevice device = VK_NULL_HANDLE;
  VkQueryPool query_pool = VK_NULL_HANDLE;
  bool enabled = false;

  TimestampQuery() = default;
  ~TimestampQuery();
  TimestampQuery(const TimestampQuery&) = delete;
  TimestampQuery& operator=(const TimestampQuery&) = delete;
  TimestampQuery(TimestampQuery&& other) noexcept;
  TimestampQuery& operator=(TimestampQuery&& other) noexcept;
};

struct AutotuneSpecialization {
  std::uint32_t local_size_x = 256;
  std::uint32_t block_m = 128;
  std::uint32_t block_n = 256;
  std::uint32_t block_k = 16;
  std::uint32_t warp_m = 128;
  std::uint32_t warp_n = 32;
  std::uint32_t warp_n_iter = 1;
  std::uint32_t thread_m = 8;
  std::uint32_t thread_n = 8;
  std::uint32_t stages = 2;
};


Instance create_instance();
DeviceContext create_device(VkInstance instance);
Buffer create_host_visible_storage_buffer(const DeviceContext& device,
                                          VkDeviceSize size);
Buffer create_staging_buffer(const DeviceContext& device, VkDeviceSize size);
Buffer create_device_local_storage_buffer(const DeviceContext& device,
                                          VkDeviceSize size);
DescriptorResources create_descriptor_resources(const DeviceContext& device);
DescriptorResources create_descriptor_resources(const DeviceContext& device,
                                                std::uint32_t binding_count);
void update_descriptor_set(const DeviceContext& device,
                           const DescriptorResources& descriptors,
                           const Buffer& a, const Buffer& b, const Buffer& c);
void update_descriptor_set(const DeviceContext& device,
                           const DescriptorResources& descriptors,
                           const std::vector<const Buffer*>& buffers);
ComputePipeline create_compute_pipeline(const DeviceContext& device,
                                        const DescriptorResources& descriptors,
                                        const std::filesystem::path& spirv_path,
                                        std::uint32_t push_constant_size,
                                        const AutotuneSpecialization* specialization);
CommandResources create_command_resources(const DeviceContext& device);
TimestampQuery create_timestamp_query(const DeviceContext& device);

void copy_to_buffer(const Buffer& buffer, const void* source,
                    std::size_t bytes);
void copy_from_buffer(const Buffer& buffer, void* destination,
                      std::size_t bytes);
void upload_to_device_buffer(const DeviceContext& device,
                             const CommandResources& commands,
                             const Buffer& staging_buffer,
                             const Buffer& device_buffer, VkDeviceSize bytes);
void download_from_device_buffer(const DeviceContext& device,
                                 const CommandResources& commands,
                                 const Buffer& device_buffer,
                                 const Buffer& staging_buffer,
                                 VkDeviceSize bytes);
std::vector<std::uint32_t> read_spirv_file(const std::filesystem::path& path);

}  // namespace gemm::runtime
