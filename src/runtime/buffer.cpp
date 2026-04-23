#include <cstring>
#include <stdexcept>
#include <utility>

#include "runtime/runtime.hpp"

namespace gemm::runtime {

namespace {

std::uint32_t find_memory_type(VkPhysicalDevice physical_device,
                               std::uint32_t type_filter,
                               VkMemoryPropertyFlags properties) {
  VkPhysicalDeviceMemoryProperties memory_properties{};
  vkGetPhysicalDeviceMemoryProperties(physical_device, &memory_properties);

  for (std::uint32_t index = 0; index < memory_properties.memoryTypeCount;
       ++index) {
    const bool type_matches = (type_filter & (1U << index)) != 0;
    const bool property_matches =
        (memory_properties.memoryTypes[index].propertyFlags & properties) ==
        properties;
    if (type_matches && property_matches) {
      return index;
    }
  }

  throw std::runtime_error("Unable to find a matching Vulkan memory type");
}

void submit_and_wait(const DeviceContext& device,
                     const CommandResources& commands) {
  VkFence fence = VK_NULL_HANDLE;
  VkFenceCreateInfo fence_info{};
  fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
  check_vk(vkCreateFence(device.device, &fence_info, nullptr, &fence),
           "vkCreateFence");

  VkSubmitInfo submit_info{};
  submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  submit_info.commandBufferCount = 1;
  submit_info.pCommandBuffers = &commands.command_buffer;

  check_vk(vkQueueSubmit(device.compute_queue, 1, &submit_info, fence),
           "vkQueueSubmit");
  check_vk(vkWaitForFences(device.device, 1, &fence, VK_TRUE, UINT64_MAX),
           "vkWaitForFences");
  vkDestroyFence(device.device, fence, nullptr);
}

Buffer create_buffer(const DeviceContext& device_context, VkDeviceSize size,
                     VkBufferUsageFlags usage,
                     VkMemoryPropertyFlags memory_properties, bool map_memory) {
  Buffer buffer{};
  buffer.device = device_context.device;
  buffer.size = size;

  VkBufferCreateInfo buffer_info{};
  buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  buffer_info.size = size;
  buffer_info.usage = usage;
  buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

  check_vk(vkCreateBuffer(device_context.device, &buffer_info, nullptr,
                          &buffer.buffer),
           "vkCreateBuffer");

  VkMemoryRequirements requirements{};
  vkGetBufferMemoryRequirements(device_context.device, buffer.buffer,
                                &requirements);

  VkMemoryAllocateInfo alloc_info{};
  alloc_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
  alloc_info.allocationSize = requirements.size;
  alloc_info.memoryTypeIndex =
      find_memory_type(device_context.physical_device,
                       requirements.memoryTypeBits, memory_properties);

  check_vk(vkAllocateMemory(device_context.device, &alloc_info, nullptr,
                            &buffer.memory),
           "vkAllocateMemory");
  check_vk(vkBindBufferMemory(device_context.device, buffer.buffer,
                              buffer.memory, 0),
           "vkBindBufferMemory");

  if (map_memory) {
    check_vk(vkMapMemory(device_context.device, buffer.memory, 0, size, 0,
                         &buffer.mapped),
             "vkMapMemory");
  }

  return buffer;
}

}  // namespace

Buffer::~Buffer() {
  if (mapped != nullptr && device != VK_NULL_HANDLE &&
      memory != VK_NULL_HANDLE) {
    vkUnmapMemory(device, memory);
  }
  if (buffer != VK_NULL_HANDLE && device != VK_NULL_HANDLE) {
    vkDestroyBuffer(device, buffer, nullptr);
  }
  if (memory != VK_NULL_HANDLE && device != VK_NULL_HANDLE) {
    vkFreeMemory(device, memory, nullptr);
  }
}

Buffer::Buffer(Buffer&& other) noexcept
    : device(std::exchange(other.device, VK_NULL_HANDLE)),
      buffer(std::exchange(other.buffer, VK_NULL_HANDLE)),
      memory(std::exchange(other.memory, VK_NULL_HANDLE)),
      mapped(std::exchange(other.mapped, nullptr)),
      size(std::exchange(other.size, 0)) {}

Buffer& Buffer::operator=(Buffer&& other) noexcept {
  if (this != &other) {
    if (mapped != nullptr && device != VK_NULL_HANDLE &&
        memory != VK_NULL_HANDLE) {
      vkUnmapMemory(device, memory);
    }
    if (buffer != VK_NULL_HANDLE && device != VK_NULL_HANDLE) {
      vkDestroyBuffer(device, buffer, nullptr);
    }
    if (memory != VK_NULL_HANDLE && device != VK_NULL_HANDLE) {
      vkFreeMemory(device, memory, nullptr);
    }
    device = std::exchange(other.device, VK_NULL_HANDLE);
    buffer = std::exchange(other.buffer, VK_NULL_HANDLE);
    memory = std::exchange(other.memory, VK_NULL_HANDLE);
    mapped = std::exchange(other.mapped, nullptr);
    size = std::exchange(other.size, 0);
  }
  return *this;
}

Buffer create_host_visible_storage_buffer(const DeviceContext& device_context,
                                          VkDeviceSize size) {
  return create_buffer(device_context, size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                       VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                           VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                       true);
}

Buffer create_staging_buffer(const DeviceContext& device_context,
                             VkDeviceSize size) {
  return create_buffer(
      device_context, size,
      VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
          VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
      true);
}

Buffer create_device_local_storage_buffer(const DeviceContext& device_context,
                                          VkDeviceSize size) {
  return create_buffer(device_context, size,
                       VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                           VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                           VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                       VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, false);
}

void copy_to_buffer(const Buffer& buffer, const void* source,
                    std::size_t bytes) {
  if (buffer.mapped == nullptr) {
    throw std::runtime_error("copy_to_buffer requires a mapped buffer");
  }
  if (bytes > static_cast<std::size_t>(buffer.size)) {
    throw std::runtime_error("copy_to_buffer exceeds buffer size");
  }
  std::memcpy(buffer.mapped, source, bytes);
}

void copy_from_buffer(const Buffer& buffer, void* destination,
                      std::size_t bytes) {
  if (buffer.mapped == nullptr) {
    throw std::runtime_error("copy_from_buffer requires a mapped buffer");
  }
  if (bytes > static_cast<std::size_t>(buffer.size)) {
    throw std::runtime_error("copy_from_buffer exceeds buffer size");
  }
  std::memcpy(destination, buffer.mapped, bytes);
}

void upload_to_device_buffer(const DeviceContext& device,
                             const CommandResources& commands,
                             const Buffer& staging_buffer,
                             const Buffer& device_buffer, VkDeviceSize bytes) {
  check_vk(vkResetCommandPool(device.device, commands.command_pool, 0),
           "vkResetCommandPool");

  VkCommandBufferBeginInfo begin_info{};
  begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  check_vk(vkBeginCommandBuffer(commands.command_buffer, &begin_info),
           "vkBeginCommandBuffer");

  VkBufferCopy copy_region{};
  copy_region.size = bytes;
  vkCmdCopyBuffer(commands.command_buffer, staging_buffer.buffer,
                  device_buffer.buffer, 1, &copy_region);

  VkBufferMemoryBarrier barrier{};
  barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
  barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
  barrier.dstAccessMask =
      VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
  barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  barrier.buffer = device_buffer.buffer;
  barrier.offset = 0;
  barrier.size = bytes;
  vkCmdPipelineBarrier(commands.command_buffer, VK_PIPELINE_STAGE_TRANSFER_BIT,
                       VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 1,
                       &barrier, 0, nullptr);

  check_vk(vkEndCommandBuffer(commands.command_buffer), "vkEndCommandBuffer");
  submit_and_wait(device, commands);
}

void download_from_device_buffer(const DeviceContext& device,
                                 const CommandResources& commands,
                                 const Buffer& device_buffer,
                                 const Buffer& staging_buffer,
                                 VkDeviceSize bytes) {
  check_vk(vkResetCommandPool(device.device, commands.command_pool, 0),
           "vkResetCommandPool");

  VkCommandBufferBeginInfo begin_info{};
  begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  check_vk(vkBeginCommandBuffer(commands.command_buffer, &begin_info),
           "vkBeginCommandBuffer");

  VkBufferMemoryBarrier before_copy{};
  before_copy.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
  before_copy.srcAccessMask =
      VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
  before_copy.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
  before_copy.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  before_copy.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  before_copy.buffer = device_buffer.buffer;
  before_copy.offset = 0;
  before_copy.size = bytes;
  vkCmdPipelineBarrier(commands.command_buffer,
                       VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                       VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 1,
                       &before_copy, 0, nullptr);

  VkBufferCopy copy_region{};
  copy_region.size = bytes;
  vkCmdCopyBuffer(commands.command_buffer, device_buffer.buffer,
                  staging_buffer.buffer, 1, &copy_region);

  VkBufferMemoryBarrier after_copy{};
  after_copy.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
  after_copy.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
  after_copy.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
  after_copy.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  after_copy.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  after_copy.buffer = staging_buffer.buffer;
  after_copy.offset = 0;
  after_copy.size = bytes;
  vkCmdPipelineBarrier(commands.command_buffer, VK_PIPELINE_STAGE_TRANSFER_BIT,
                       VK_PIPELINE_STAGE_HOST_BIT, 0, 0, nullptr, 1,
                       &after_copy, 0, nullptr);

  check_vk(vkEndCommandBuffer(commands.command_buffer), "vkEndCommandBuffer");
  submit_and_wait(device, commands);
}

}  // namespace gemm::runtime
