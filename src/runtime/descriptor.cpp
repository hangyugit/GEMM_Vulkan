#include <array>
#include <utility>
#include <vector>

#include "runtime/runtime.hpp"

namespace gemm::runtime {

DescriptorResources::~DescriptorResources() {
  if (pool != VK_NULL_HANDLE && device != VK_NULL_HANDLE) {
    vkDestroyDescriptorPool(device, pool, nullptr);
  }
  if (layout != VK_NULL_HANDLE && device != VK_NULL_HANDLE) {
    vkDestroyDescriptorSetLayout(device, layout, nullptr);
  }
}

DescriptorResources::DescriptorResources(DescriptorResources&& other) noexcept
    : device(std::exchange(other.device, VK_NULL_HANDLE)),
      layout(std::exchange(other.layout, VK_NULL_HANDLE)),
      pool(std::exchange(other.pool, VK_NULL_HANDLE)),
      set(std::exchange(other.set, VK_NULL_HANDLE)) {}

DescriptorResources& DescriptorResources::operator=(
    DescriptorResources&& other) noexcept {
  if (this != &other) {
    if (pool != VK_NULL_HANDLE && device != VK_NULL_HANDLE) {
      vkDestroyDescriptorPool(device, pool, nullptr);
    }
    if (layout != VK_NULL_HANDLE && device != VK_NULL_HANDLE) {
      vkDestroyDescriptorSetLayout(device, layout, nullptr);
    }
    device = std::exchange(other.device, VK_NULL_HANDLE);
    layout = std::exchange(other.layout, VK_NULL_HANDLE);
    pool = std::exchange(other.pool, VK_NULL_HANDLE);
    set = std::exchange(other.set, VK_NULL_HANDLE);
  }
  return *this;
}

DescriptorResources create_descriptor_resources(const DeviceContext& device,
                                                std::uint32_t binding_count) {
  DescriptorResources descriptors{};
  descriptors.device = device.device;

  std::vector<VkDescriptorSetLayoutBinding> bindings(binding_count);
  for (std::uint32_t index = 0; index < bindings.size(); ++index) {
    bindings[index].binding = index;
    bindings[index].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[index].descriptorCount = 1;
    bindings[index].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
  }

  VkDescriptorSetLayoutCreateInfo layout_info{};
  layout_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
  layout_info.bindingCount = static_cast<std::uint32_t>(bindings.size());
  layout_info.pBindings = bindings.data();
  check_vk(vkCreateDescriptorSetLayout(device.device, &layout_info, nullptr,
                                       &descriptors.layout),
           "vkCreateDescriptorSetLayout");

  VkDescriptorPoolSize pool_size{};
  pool_size.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  pool_size.descriptorCount = static_cast<std::uint32_t>(bindings.size());

  VkDescriptorPoolCreateInfo pool_info{};
  pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
  pool_info.maxSets = 1;
  pool_info.poolSizeCount = 1;
  pool_info.pPoolSizes = &pool_size;
  check_vk(vkCreateDescriptorPool(device.device, &pool_info, nullptr,
                                  &descriptors.pool),
           "vkCreateDescriptorPool");

  VkDescriptorSetAllocateInfo alloc_info{};
  alloc_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
  alloc_info.descriptorPool = descriptors.pool;
  alloc_info.descriptorSetCount = 1;
  alloc_info.pSetLayouts = &descriptors.layout;
  check_vk(
      vkAllocateDescriptorSets(device.device, &alloc_info, &descriptors.set),
      "vkAllocateDescriptorSets");

  return descriptors;
}

DescriptorResources create_descriptor_resources(const DeviceContext& device) {
  return create_descriptor_resources(device, 3);
}

void update_descriptor_set(const DeviceContext& device,
                           const DescriptorResources& descriptors,
                           const Buffer& a, const Buffer& b, const Buffer& c) {
  update_descriptor_set(device, descriptors, std::vector<const Buffer*>{&a, &b, &c});
}

void update_descriptor_set(const DeviceContext& device,
                           const DescriptorResources& descriptors,
                           const std::vector<const Buffer*>& buffers) {
  std::vector<VkDescriptorBufferInfo> buffer_infos(buffers.size());
  for (std::uint32_t index = 0; index < buffers.size(); ++index) {
    buffer_infos[index] =
        VkDescriptorBufferInfo{buffers[index]->buffer, 0, buffers[index]->size};
  }

  std::vector<VkWriteDescriptorSet> writes(buffers.size());
  for (std::uint32_t index = 0; index < writes.size(); ++index) {
    writes[index].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[index].dstSet = descriptors.set;
    writes[index].dstBinding = index;
    writes[index].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[index].descriptorCount = 1;
    writes[index].pBufferInfo = &buffer_infos[index];
  }

  vkUpdateDescriptorSets(device.device,
                         static_cast<std::uint32_t>(writes.size()),
                         writes.data(), 0, nullptr);
}

}  // namespace gemm::runtime
