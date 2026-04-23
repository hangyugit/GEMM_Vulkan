#include <stdexcept>
#include <string>
#include <utility>

#include "runtime/runtime.hpp"

namespace gemm::runtime {

namespace {

VkApplicationInfo make_application_info() {
  VkApplicationInfo info{};
  info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
  info.pApplicationName = "gemm_vulkan";
  info.applicationVersion = VK_MAKE_VERSION(0, 1, 0);
  info.pEngineName = "none";
  info.engineVersion = VK_MAKE_VERSION(0, 1, 0);
  info.apiVersion = VK_API_VERSION_1_2;
  return info;
}

}  // namespace

void check_vk(VkResult result, std::string_view message) {
  if (result != VK_SUCCESS) {
    throw std::runtime_error(std::string(message) + " failed with code " +
                             std::to_string(result));
  }
}

Instance::~Instance() {
  if (handle != VK_NULL_HANDLE) {
    vkDestroyInstance(handle, nullptr);
  }
}

Instance::Instance(Instance&& other) noexcept
    : handle(std::exchange(other.handle, VK_NULL_HANDLE)) {}

Instance& Instance::operator=(Instance&& other) noexcept {
  if (this != &other) {
    if (handle != VK_NULL_HANDLE) {
      vkDestroyInstance(handle, nullptr);
    }
    handle = std::exchange(other.handle, VK_NULL_HANDLE);
  }
  return *this;
}

Instance create_instance() {
  Instance instance{};

  VkApplicationInfo app_info = make_application_info();
  VkInstanceCreateInfo create_info{};
  create_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
  create_info.pApplicationInfo = &app_info;

  check_vk(vkCreateInstance(&create_info, nullptr, &instance.handle),
           "vkCreateInstance");
  return instance;
}

}  // namespace gemm::runtime
