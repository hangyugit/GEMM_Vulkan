#include <utility>

#include "runtime/runtime.hpp"

namespace gemm::runtime {

CommandResources::~CommandResources() {
  if (command_pool != VK_NULL_HANDLE && device != VK_NULL_HANDLE) {
    vkDestroyCommandPool(device, command_pool, nullptr);
  }
}

CommandResources::CommandResources(CommandResources&& other) noexcept
    : device(std::exchange(other.device, VK_NULL_HANDLE)),
      command_pool(std::exchange(other.command_pool, VK_NULL_HANDLE)),
      command_buffer(std::exchange(other.command_buffer, VK_NULL_HANDLE)) {}

CommandResources& CommandResources::operator=(
    CommandResources&& other) noexcept {
  if (this != &other) {
    if (command_pool != VK_NULL_HANDLE && device != VK_NULL_HANDLE) {
      vkDestroyCommandPool(device, command_pool, nullptr);
    }
    device = std::exchange(other.device, VK_NULL_HANDLE);
    command_pool = std::exchange(other.command_pool, VK_NULL_HANDLE);
    command_buffer = std::exchange(other.command_buffer, VK_NULL_HANDLE);
  }
  return *this;
}

CommandResources create_command_resources(const DeviceContext& device) {
  CommandResources resources{};
  resources.device = device.device;

  VkCommandPoolCreateInfo pool_info{};
  pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
  pool_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
  pool_info.queueFamilyIndex = device.queue_family_index;
  check_vk(vkCreateCommandPool(device.device, &pool_info, nullptr,
                               &resources.command_pool),
           "vkCreateCommandPool");

  VkCommandBufferAllocateInfo alloc_info{};
  alloc_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
  alloc_info.commandPool = resources.command_pool;
  alloc_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  alloc_info.commandBufferCount = 1;
  check_vk(vkAllocateCommandBuffers(device.device, &alloc_info,
                                    &resources.command_buffer),
           "vkAllocateCommandBuffers");

  return resources;
}

}  // namespace gemm::runtime
