#include <utility>

#include "runtime/runtime.hpp"

namespace gemm::runtime {

TimestampQuery::~TimestampQuery() {
  if (query_pool != VK_NULL_HANDLE && device != VK_NULL_HANDLE) {
    vkDestroyQueryPool(device, query_pool, nullptr);
  }
}

TimestampQuery::TimestampQuery(TimestampQuery&& other) noexcept
    : device(std::exchange(other.device, VK_NULL_HANDLE)),
      query_pool(std::exchange(other.query_pool, VK_NULL_HANDLE)),
      enabled(std::exchange(other.enabled, false)) {}

TimestampQuery& TimestampQuery::operator=(TimestampQuery&& other) noexcept {
  if (this != &other) {
    if (query_pool != VK_NULL_HANDLE && device != VK_NULL_HANDLE) {
      vkDestroyQueryPool(device, query_pool, nullptr);
    }
    device = std::exchange(other.device, VK_NULL_HANDLE);
    query_pool = std::exchange(other.query_pool, VK_NULL_HANDLE);
    enabled = std::exchange(other.enabled, false);
  }
  return *this;
}

TimestampQuery create_timestamp_query(const DeviceContext& device) {
  TimestampQuery query{};
  query.device = device.device;
  query.enabled = device.supports_timestamps;

  if (!query.enabled) {
    return query;
  }

  VkQueryPoolCreateInfo pool_info{};
  pool_info.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
  pool_info.queryType = VK_QUERY_TYPE_TIMESTAMP;
  pool_info.queryCount = 2;
  check_vk(
      vkCreateQueryPool(device.device, &pool_info, nullptr, &query.query_pool),
      "vkCreateQueryPool");
  return query;
}

}  // namespace gemm::runtime
