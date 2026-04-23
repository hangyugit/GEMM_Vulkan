#include <algorithm>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

#include "runtime/runtime.hpp"

namespace gemm::runtime {

namespace {

struct DeviceSelection {
  VkPhysicalDevice physical_device = VK_NULL_HANDLE;
  std::uint32_t queue_family_index = 0;
};

int score_device(VkPhysicalDevice device,
                 const std::vector<VkQueueFamilyProperties>& families,
                 std::uint32_t* queue_family_index) {
  VkPhysicalDeviceProperties properties{};
  vkGetPhysicalDeviceProperties(device, &properties);

  bool has_compute = false;
  std::uint32_t compute_index = 0;
  bool has_dedicated_compute = false;

  for (std::uint32_t index = 0; index < families.size(); ++index) {
    if ((families[index].queueFlags & VK_QUEUE_COMPUTE_BIT) == 0) {
      continue;
    }

    has_compute = true;
    compute_index = index;
    if ((families[index].queueFlags & VK_QUEUE_GRAPHICS_BIT) == 0) {
      has_dedicated_compute = true;
      break;
    }
  }

  if (!has_compute) {
    return std::numeric_limits<int>::min();
  }

  *queue_family_index = compute_index;

  int score = 0;
  if (properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
    score += 1000;
  }
  if (has_dedicated_compute) {
    score += 200;
  }
  score +=
      static_cast<int>(properties.limits.maxComputeSharedMemorySize / 1024);
  return score;
}

DeviceSelection pick_physical_device(VkInstance instance) {
  std::uint32_t device_count = 0;
  check_vk(vkEnumeratePhysicalDevices(instance, &device_count, nullptr),
           "vkEnumeratePhysicalDevices(count)");
  if (device_count == 0) {
    throw std::runtime_error("No Vulkan physical devices found");
  }

  std::vector<VkPhysicalDevice> devices(device_count);
  check_vk(vkEnumeratePhysicalDevices(instance, &device_count, devices.data()),
           "vkEnumeratePhysicalDevices(list)");

  int best_score = std::numeric_limits<int>::min();
  DeviceSelection best{};
  for (VkPhysicalDevice device : devices) {
    std::uint32_t queue_family_count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(device, &queue_family_count,
                                             nullptr);
    std::vector<VkQueueFamilyProperties> families(queue_family_count);
    vkGetPhysicalDeviceQueueFamilyProperties(device, &queue_family_count,
                                             families.data());

    std::uint32_t queue_family_index = 0;
    const int score = score_device(device, families, &queue_family_index);
    if (score > best_score) {
      best_score = score;
      best.physical_device = device;
      best.queue_family_index = queue_family_index;
    }
  }

  if (best.physical_device == VK_NULL_HANDLE) {
    throw std::runtime_error("No Vulkan device with a compute queue was found");
  }

  return best;
}

}  // namespace

DeviceContext::~DeviceContext() {
  if (device != VK_NULL_HANDLE) {
    vkDestroyDevice(device, nullptr);
  }
}

DeviceContext::DeviceContext(DeviceContext&& other) noexcept
    : physical_device(std::exchange(other.physical_device, VK_NULL_HANDLE)),
      device(std::exchange(other.device, VK_NULL_HANDLE)),
      compute_queue(std::exchange(other.compute_queue, VK_NULL_HANDLE)),
      queue_family_index(other.queue_family_index),
      timestamp_period(other.timestamp_period),
      supports_timestamps(other.supports_timestamps) {}

DeviceContext& DeviceContext::operator=(DeviceContext&& other) noexcept {
  if (this != &other) {
    if (device != VK_NULL_HANDLE) {
      vkDestroyDevice(device, nullptr);
    }
    physical_device = std::exchange(other.physical_device, VK_NULL_HANDLE);
    device = std::exchange(other.device, VK_NULL_HANDLE);
    compute_queue = std::exchange(other.compute_queue, VK_NULL_HANDLE);
    queue_family_index = other.queue_family_index;
    timestamp_period = other.timestamp_period;
    supports_timestamps = other.supports_timestamps;
  }
  return *this;
}

DeviceContext create_device(VkInstance instance) {
  const DeviceSelection selection = pick_physical_device(instance);

  float queue_priority = 1.0F;
  VkDeviceQueueCreateInfo queue_info{};
  queue_info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
  queue_info.queueFamilyIndex = selection.queue_family_index;
  queue_info.queueCount = 1;
  queue_info.pQueuePriorities = &queue_priority;

  VkPhysicalDeviceFeatures device_features{};
  VkDeviceCreateInfo create_info{};
  create_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
  create_info.queueCreateInfoCount = 1;
  create_info.pQueueCreateInfos = &queue_info;
  create_info.pEnabledFeatures = &device_features;

  DeviceContext context{};
  context.physical_device = selection.physical_device;
  context.queue_family_index = selection.queue_family_index;

  check_vk(vkCreateDevice(selection.physical_device, &create_info, nullptr,
                          &context.device),
           "vkCreateDevice");
  vkGetDeviceQueue(context.device, context.queue_family_index, 0,
                   &context.compute_queue);

  VkPhysicalDeviceProperties properties{};
  vkGetPhysicalDeviceProperties(context.physical_device, &properties);
  context.timestamp_period = properties.limits.timestampPeriod;

  std::uint32_t queue_family_count = 0;
  vkGetPhysicalDeviceQueueFamilyProperties(context.physical_device,
                                           &queue_family_count, nullptr);
  std::vector<VkQueueFamilyProperties> families(queue_family_count);
  vkGetPhysicalDeviceQueueFamilyProperties(
      context.physical_device, &queue_family_count, families.data());
  context.supports_timestamps =
      families[context.queue_family_index].timestampValidBits > 0;

  return context;
}

}  // namespace gemm::runtime
