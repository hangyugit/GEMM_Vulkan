#include "gemm/runner.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "gemm/internal.hpp"
#include "gemm/kernels/launch_config.hpp"
#include "gemm/kernels/registry.hpp"
#include "runtime/runtime.hpp"

namespace gemm {

namespace {

struct PushConstants {
  std::uint32_t m = 0;
  std::uint32_t n = 0;
  std::uint32_t k = 0;
  float alpha = 1.0F;
  float beta = 0.0F;
};

struct AdvancedPushConstants {
  std::uint32_t m = 0;
  std::uint32_t n = 0;
  std::uint32_t k = 0;
  float alpha = 1.0F;
  float beta = 0.0F;
  std::uint32_t split_count = 1;
  std::uint32_t total_tiles = 0;
};

std::uint32_t ceil_div(std::uint32_t value, std::uint32_t divisor) {
  return (value + divisor - 1) / divisor;
}

SampleStats summarize_samples(const std::vector<double>& samples) {
  SampleStats stats{};
  if (samples.empty()) {
    return stats;
  }

  const auto [minimum, maximum] =
      std::minmax_element(samples.begin(), samples.end());
  const double sum = std::accumulate(samples.begin(), samples.end(), 0.0);
  const double average = sum / static_cast<double>(samples.size());

  double squared_error_sum = 0.0;
  for (double sample : samples) {
    const double delta = sample - average;
    squared_error_sum += delta * delta;
  }

  stats.average = average;
  stats.minimum = *minimum;
  stats.maximum = *maximum;
  stats.stddev =
      std::sqrt(squared_error_sum / static_cast<double>(samples.size()));
  return stats;
}

void validate_problem(const ProblemSize& problem) {
  if (problem.m == 0 || problem.n == 0 || problem.k == 0) {
    throw std::runtime_error(
        "Problem dimensions must all be greater than zero");
  }
}

KernelConfig make_effective_config(const KernelDefinition& kernel,
                                   const RunOptions& options) {
  KernelConfig config = kernel.config;
  if (!options.autotune.has_value()) {
    return config;
  }
  if (kernel.info.id != 13) {
    throw std::runtime_error("--spec-* options are only supported by kernel 13");
  }

  const AutotuneOptions& autotune = *options.autotune;
  config.workgroup_x = autotune.local_size_x;
  config.workgroup_y = 1;
  config.workgroup_z = 1;
  config.block_m = autotune.block_m;
  config.block_n = autotune.block_n;
  config.block_k = autotune.block_k;
  config.thread_m = autotune.thread_m;
  config.thread_n = autotune.thread_n;
  config.vector_width = 4;
  config.use_shared_memory = true;
  config.use_subgroup = true;
  config.dispatch_order = autotune.dispatch_order;
  return config;
}

runtime::AutotuneSpecialization make_specialization(
    const KernelConfig& config, const std::optional<AutotuneOptions>& override) {
  runtime::AutotuneSpecialization specialization{
      .local_size_x = config.workgroup_x,
      .block_m = config.block_m,
      .block_n = config.block_n,
      .block_k = config.block_k,
      .warp_m = 128,
      .warp_n = 32,
      .warp_n_iter = 1,
      .thread_m = config.thread_m,
      .thread_n = config.thread_n,
      .stages = 2,
  };

  if (override.has_value()) {
    const AutotuneOptions& autotune = *override;
    specialization.warp_m = autotune.warp_m;
    specialization.warp_n = autotune.warp_n;
    specialization.warp_n_iter = autotune.warp_n_iter;
    specialization.stages = autotune.stages;
  }
  return specialization;
}

double run_dispatch_once(const RunOptions& options,
                         const runtime::DeviceContext& device,
                         const runtime::Buffer& c_buffer,
                         const runtime::Buffer& c_staging,
                         const std::vector<float>& c_initial,
                         const runtime::DescriptorResources& descriptors,
                         const runtime::ComputePipeline& pipeline,
                         const runtime::CommandResources& commands,
                         const runtime::TimestampQuery& timestamps,
                         const DispatchShape& dispatch, bool collect_profile,
                         double* wall_ms_out) {
  runtime::copy_to_buffer(c_staging, c_initial.data(),
                          c_initial.size() * sizeof(float));
  runtime::upload_to_device_buffer(device, commands, c_staging, c_buffer,
                                   c_initial.size() * sizeof(float));

  runtime::check_vk(vkResetCommandPool(device.device, commands.command_pool, 0),
                    "vkResetCommandPool");

  VkCommandBufferBeginInfo begin_info{};
  begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  runtime::check_vk(vkBeginCommandBuffer(commands.command_buffer, &begin_info),
                    "vkBeginCommandBuffer");

  if (collect_profile && timestamps.enabled) {
    vkCmdResetQueryPool(commands.command_buffer, timestamps.query_pool, 0, 2);
    vkCmdWriteTimestamp(commands.command_buffer,
                        VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                        timestamps.query_pool, 0);
  }

  vkCmdBindPipeline(commands.command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                    pipeline.pipeline);
  vkCmdBindDescriptorSets(
      commands.command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE,
      pipeline.pipeline_layout, 0, 1, &descriptors.set, 0, nullptr);

  const PushConstants push_constants{
      .m = options.problem.m,
      .n = options.problem.n,
      .k = options.problem.k,
      .alpha = options.scalars.alpha,
      .beta = options.scalars.beta,
  };
  vkCmdPushConstants(commands.command_buffer, pipeline.pipeline_layout,
                     VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PushConstants),
                     &push_constants);

  vkCmdDispatch(commands.command_buffer, dispatch.x, dispatch.y, dispatch.z);

  if (collect_profile && timestamps.enabled) {
    vkCmdWriteTimestamp(commands.command_buffer,
                        VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
                        timestamps.query_pool, 1);
  }

  runtime::check_vk(vkEndCommandBuffer(commands.command_buffer),
                    "vkEndCommandBuffer");

  VkFence fence = VK_NULL_HANDLE;
  VkFenceCreateInfo fence_info{};
  fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
  runtime::check_vk(vkCreateFence(device.device, &fence_info, nullptr, &fence),
                    "vkCreateFence");

  VkSubmitInfo submit_info{};
  submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  submit_info.commandBufferCount = 1;
  submit_info.pCommandBuffers = &commands.command_buffer;

  const auto wall_start = std::chrono::steady_clock::now();
  runtime::check_vk(vkQueueSubmit(device.compute_queue, 1, &submit_info, fence),
                    "vkQueueSubmit");
  runtime::check_vk(
      vkWaitForFences(device.device, 1, &fence, VK_TRUE, UINT64_MAX),
      "vkWaitForFences");
  const auto wall_end = std::chrono::steady_clock::now();
  vkDestroyFence(device.device, fence, nullptr);

  *wall_ms_out =
      std::chrono::duration<double, std::milli>(wall_end - wall_start).count();

  if (!(collect_profile && timestamps.enabled)) {
    return *wall_ms_out;
  }

  std::array<std::uint64_t, 2> results{};
  runtime::check_vk(vkGetQueryPoolResults(
                        device.device, timestamps.query_pool, 0, 2,
                        sizeof(results), results.data(), sizeof(std::uint64_t),
                        VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT),
                    "vkGetQueryPoolResults");

  return static_cast<double>(results[1] - results[0]) *
         static_cast<double>(device.timestamp_period) / 1.0e6;
}

double run_dispatch_batch(const RunOptions& options,
                          const runtime::DeviceContext& device,
                          const runtime::Buffer& c_buffer,
                          const runtime::Buffer& c_staging,
                          const std::vector<float>& c_initial,
                          const runtime::DescriptorResources& descriptors,
                          const runtime::ComputePipeline& pipeline,
                          const runtime::CommandResources& commands,
                          const runtime::TimestampQuery& timestamps,
                          const DispatchShape& dispatch,
                          std::uint32_t dispatch_count, bool collect_profile,
                          double* average_wall_ms_out) {
  if (dispatch_count == 0) {
    throw std::runtime_error("dispatch_count must be greater than zero");
  }

  runtime::copy_to_buffer(c_staging, c_initial.data(),
                          c_initial.size() * sizeof(float));
  runtime::upload_to_device_buffer(device, commands, c_staging, c_buffer,
                                   c_initial.size() * sizeof(float));

  runtime::check_vk(vkResetCommandPool(device.device, commands.command_pool, 0),
                    "vkResetCommandPool");

  VkCommandBufferBeginInfo begin_info{};
  begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  runtime::check_vk(vkBeginCommandBuffer(commands.command_buffer, &begin_info),
                    "vkBeginCommandBuffer");

  if (collect_profile && timestamps.enabled) {
    vkCmdResetQueryPool(commands.command_buffer, timestamps.query_pool, 0, 2);
    vkCmdWriteTimestamp(commands.command_buffer,
                        VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                        timestamps.query_pool, 0);
  }

  vkCmdBindPipeline(commands.command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                    pipeline.pipeline);
  vkCmdBindDescriptorSets(
      commands.command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE,
      pipeline.pipeline_layout, 0, 1, &descriptors.set, 0, nullptr);

  const PushConstants push_constants{
      .m = options.problem.m,
      .n = options.problem.n,
      .k = options.problem.k,
      .alpha = options.scalars.alpha,
      .beta = options.scalars.beta,
  };
  vkCmdPushConstants(commands.command_buffer, pipeline.pipeline_layout,
                     VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PushConstants),
                     &push_constants);

  for (std::uint32_t iter = 0; iter < dispatch_count; ++iter) {
    vkCmdDispatch(commands.command_buffer, dispatch.x, dispatch.y, dispatch.z);

    if (iter + 1 < dispatch_count) {
      VkBufferMemoryBarrier barrier{};
      barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
      barrier.srcAccessMask =
          VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
      barrier.dstAccessMask =
          VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
      barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
      barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
      barrier.buffer = c_buffer.buffer;
      barrier.offset = 0;
      barrier.size = c_buffer.size;
      vkCmdPipelineBarrier(commands.command_buffer,
                           VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                           VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr,
                           1, &barrier, 0, nullptr);
    }
  }

  if (collect_profile && timestamps.enabled) {
    vkCmdWriteTimestamp(commands.command_buffer,
                        VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
                        timestamps.query_pool, 1);
  }

  runtime::check_vk(vkEndCommandBuffer(commands.command_buffer),
                    "vkEndCommandBuffer");

  VkFence fence = VK_NULL_HANDLE;
  VkFenceCreateInfo fence_info{};
  fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
  runtime::check_vk(vkCreateFence(device.device, &fence_info, nullptr, &fence),
                    "vkCreateFence");

  VkSubmitInfo submit_info{};
  submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  submit_info.commandBufferCount = 1;
  submit_info.pCommandBuffers = &commands.command_buffer;

  const auto wall_start = std::chrono::steady_clock::now();
  runtime::check_vk(vkQueueSubmit(device.compute_queue, 1, &submit_info, fence),
                    "vkQueueSubmit");
  runtime::check_vk(
      vkWaitForFences(device.device, 1, &fence, VK_TRUE, UINT64_MAX),
      "vkWaitForFences");
  const auto wall_end = std::chrono::steady_clock::now();
  vkDestroyFence(device.device, fence, nullptr);

  const double total_wall_ms =
      std::chrono::duration<double, std::milli>(wall_end - wall_start).count();
  *average_wall_ms_out = total_wall_ms / static_cast<double>(dispatch_count);

  if (!(collect_profile && timestamps.enabled)) {
    return *average_wall_ms_out;
  }

  std::array<std::uint64_t, 2> results{};
  runtime::check_vk(vkGetQueryPoolResults(
                        device.device, timestamps.query_pool, 0, 2,
                        sizeof(results), results.data(), sizeof(std::uint64_t),
                        VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT),
                    "vkGetQueryPoolResults");

  const double total_kernel_ms = static_cast<double>(results[1] - results[0]) *
                                 static_cast<double>(device.timestamp_period) /
                                 1.0e6;
  return total_kernel_ms / static_cast<double>(dispatch_count);
}

template <typename RecordCommands>
double record_submit_timed(const runtime::DeviceContext& device,
                           const runtime::CommandResources& commands,
                           const runtime::TimestampQuery& timestamps,
                           bool collect_profile, RecordCommands record_commands,
                           double* wall_ms_out) {
  runtime::check_vk(vkResetCommandPool(device.device, commands.command_pool, 0),
                    "vkResetCommandPool");

  VkCommandBufferBeginInfo begin_info{};
  begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  runtime::check_vk(vkBeginCommandBuffer(commands.command_buffer, &begin_info),
                    "vkBeginCommandBuffer");

  if (collect_profile && timestamps.enabled) {
    vkCmdResetQueryPool(commands.command_buffer, timestamps.query_pool, 0, 2);
    vkCmdWriteTimestamp(commands.command_buffer,
                        VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                        timestamps.query_pool, 0);
  }

  record_commands(commands.command_buffer);

  if (collect_profile && timestamps.enabled) {
    vkCmdWriteTimestamp(commands.command_buffer,
                        VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
                        timestamps.query_pool, 1);
  }

  runtime::check_vk(vkEndCommandBuffer(commands.command_buffer),
                    "vkEndCommandBuffer");

  VkFence fence = VK_NULL_HANDLE;
  VkFenceCreateInfo fence_info{};
  fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
  runtime::check_vk(vkCreateFence(device.device, &fence_info, nullptr, &fence),
                    "vkCreateFence");

  VkSubmitInfo submit_info{};
  submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  submit_info.commandBufferCount = 1;
  submit_info.pCommandBuffers = &commands.command_buffer;

  const auto wall_start = std::chrono::steady_clock::now();
  runtime::check_vk(vkQueueSubmit(device.compute_queue, 1, &submit_info, fence),
                    "vkQueueSubmit");
  runtime::check_vk(
      vkWaitForFences(device.device, 1, &fence, VK_TRUE, UINT64_MAX),
      "vkWaitForFences");
  const auto wall_end = std::chrono::steady_clock::now();
  vkDestroyFence(device.device, fence, nullptr);

  *wall_ms_out =
      std::chrono::duration<double, std::milli>(wall_end - wall_start).count();

  if (!(collect_profile && timestamps.enabled)) {
    return *wall_ms_out;
  }

  std::array<std::uint64_t, 2> results{};
  runtime::check_vk(vkGetQueryPoolResults(
                        device.device, timestamps.query_pool, 0, 2,
                        sizeof(results), results.data(), sizeof(std::uint64_t),
                        VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT),
                    "vkGetQueryPoolResults");

  return static_cast<double>(results[1] - results[0]) *
         static_cast<double>(device.timestamp_period) / 1.0e6;
}

}  // namespace

std::vector<KernelInfo> list_kernels() {
  std::vector<KernelInfo> infos;
  for (const KernelDefinition& definition : kernel_registry()) {
    infos.push_back(definition.info);
  }
  return infos;
}

RunResult run_advanced_kernel(const RunOptions& options,
                              const KernelDefinition& kernel) {
  if (options.split_count == 0) {
    throw std::runtime_error("--splits must be greater than zero");
  }
  if (options.scheduler_workgroups == 0) {
    throw std::runtime_error("--scheduler-workgroups must be greater than zero");
  }

  HostMatrices matrices =
      make_host_matrices(options.problem, options.benchmark.random_seed);
  if (options.benchmark.verify) {
    reference_sgemm(options.problem, options.scalars, matrices.a, matrices.b,
                    matrices.c_initial, &matrices.c_reference);
  }

  const std::size_t bytes_a = matrices.a.size() * sizeof(float);
  const std::size_t bytes_b = matrices.b.size() * sizeof(float);
  const std::size_t bytes_c = matrices.c_initial.size() * sizeof(float);
  const std::size_t bytes_partial =
      static_cast<std::size_t>(options.split_count) * matrices.c_initial.size() *
      sizeof(float);

  runtime::Instance instance = runtime::create_instance();
  runtime::DeviceContext device = runtime::create_device(instance.handle);
  runtime::CommandResources commands =
      runtime::create_command_resources(device);

  runtime::Buffer a_buffer =
      runtime::create_device_local_storage_buffer(device, bytes_a);
  runtime::Buffer b_buffer =
      runtime::create_device_local_storage_buffer(device, bytes_b);
  runtime::Buffer c_buffer =
      runtime::create_device_local_storage_buffer(device, bytes_c);
  runtime::Buffer a_staging = runtime::create_staging_buffer(device, bytes_a);
  runtime::Buffer b_staging = runtime::create_staging_buffer(device, bytes_b);
  runtime::Buffer c_staging = runtime::create_staging_buffer(device, bytes_c);

  runtime::copy_to_buffer(a_staging, matrices.a.data(), bytes_a);
  runtime::copy_to_buffer(b_staging, matrices.b.data(), bytes_b);
  runtime::copy_to_buffer(c_staging, matrices.c_initial.data(), bytes_c);
  runtime::upload_to_device_buffer(device, commands, a_staging, a_buffer,
                                   bytes_a);
  runtime::upload_to_device_buffer(device, commands, b_staging, b_buffer,
                                   bytes_b);
  runtime::upload_to_device_buffer(device, commands, c_staging, c_buffer,
                                   bytes_c);

  const bool uses_partial = kernel.info.id == 14 || kernel.info.id == 16;
  const bool uses_counter = kernel.info.id == 15 || kernel.info.id == 16;

  runtime::Buffer partial_buffer;
  if (uses_partial) {
    partial_buffer =
        runtime::create_device_local_storage_buffer(device, bytes_partial);
  }

  runtime::Buffer counter_buffer;
  runtime::Buffer counter_staging;
  if (uses_counter) {
    counter_buffer = runtime::create_device_local_storage_buffer(
        device, sizeof(std::uint32_t));
    counter_staging = runtime::create_staging_buffer(device, sizeof(std::uint32_t));
  }

  const std::filesystem::path shader_dir = kernel.shader_path.parent_path();
  const std::filesystem::path split_partial_path =
      shader_dir / "12_split_k_partial.comp.spv";
  const std::filesystem::path reduce_path =
      shader_dir / "12_split_k_reduce.comp.spv";
  const std::filesystem::path stream_partial_path =
      shader_dir / "14_stream_k_partial.comp.spv";

  std::optional<runtime::DescriptorResources> partial_descriptors;
  std::optional<runtime::DescriptorResources> reduce_descriptors;
  std::optional<runtime::DescriptorResources> scheduler_descriptors;

  std::optional<runtime::ComputePipeline> partial_pipeline;
  std::optional<runtime::ComputePipeline> reduce_pipeline;
  std::optional<runtime::ComputePipeline> scheduler_pipeline;

  if (kernel.info.id == 14) {
    partial_descriptors.emplace(runtime::create_descriptor_resources(device));
    runtime::update_descriptor_set(device, *partial_descriptors, a_buffer,
                                   b_buffer, partial_buffer);
    reduce_descriptors.emplace(runtime::create_descriptor_resources(device));
    runtime::update_descriptor_set(device, *reduce_descriptors, partial_buffer,
                                   b_buffer, c_buffer);
    partial_pipeline.emplace(runtime::create_compute_pipeline(
        device, *partial_descriptors, split_partial_path,
        sizeof(AdvancedPushConstants), nullptr));
    reduce_pipeline.emplace(runtime::create_compute_pipeline(
        device, *reduce_descriptors, reduce_path, sizeof(AdvancedPushConstants),
        nullptr));
  } else if (kernel.info.id == 15) {
    scheduler_descriptors.emplace(runtime::create_descriptor_resources(device, 4));
    runtime::update_descriptor_set(
        device, *scheduler_descriptors,
        std::vector<const runtime::Buffer*>{&a_buffer, &b_buffer, &c_buffer,
                                            &counter_buffer});
    scheduler_pipeline.emplace(runtime::create_compute_pipeline(
        device, *scheduler_descriptors, kernel.shader_path,
        sizeof(AdvancedPushConstants), nullptr));
  } else if (kernel.info.id == 16) {
    partial_descriptors.emplace(runtime::create_descriptor_resources(device, 4));
    runtime::update_descriptor_set(
        device, *partial_descriptors,
        std::vector<const runtime::Buffer*>{&a_buffer, &b_buffer, &partial_buffer,
                                            &counter_buffer});
    reduce_descriptors.emplace(runtime::create_descriptor_resources(device));
    runtime::update_descriptor_set(device, *reduce_descriptors, partial_buffer,
                                   b_buffer, c_buffer);
    partial_pipeline.emplace(runtime::create_compute_pipeline(
        device, *partial_descriptors, stream_partial_path,
        sizeof(AdvancedPushConstants), nullptr));
    reduce_pipeline.emplace(runtime::create_compute_pipeline(
        device, *reduce_descriptors, reduce_path, sizeof(AdvancedPushConstants),
        nullptr));
  }

  runtime::TimestampQuery timestamps = runtime::create_timestamp_query(device);

  const std::uint32_t tile_m_count = ceil_div(options.problem.m, 16);
  const std::uint32_t tile_n_count = ceil_div(options.problem.n, 16);
  const std::uint32_t total_tiles = tile_m_count * tile_n_count;
  const AdvancedPushConstants push_constants{
      .m = options.problem.m,
      .n = options.problem.n,
      .k = options.problem.k,
      .alpha = options.scalars.alpha,
      .beta = options.scalars.beta,
      .split_count = options.split_count,
      .total_tiles = total_tiles,
  };

  const auto reset_c = [&]() {
    runtime::copy_to_buffer(c_staging, matrices.c_initial.data(), bytes_c);
    runtime::upload_to_device_buffer(device, commands, c_staging, c_buffer,
                                     bytes_c);
  };
  const auto reset_counter = [&]() {
    if (!uses_counter) {
      return;
    }
    const std::uint32_t zero = 0;
    runtime::copy_to_buffer(counter_staging, &zero, sizeof(zero));
    runtime::upload_to_device_buffer(device, commands, counter_staging,
                                     counter_buffer, sizeof(zero));
  };

  const auto bind_pipeline = [&](VkCommandBuffer command_buffer,
                                 const runtime::ComputePipeline& pipeline,
                                 const runtime::DescriptorResources& descriptors) {
    vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                      pipeline.pipeline);
    vkCmdBindDescriptorSets(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                            pipeline.pipeline_layout, 0, 1, &descriptors.set, 0,
                            nullptr);
    vkCmdPushConstants(command_buffer, pipeline.pipeline_layout,
                       VK_SHADER_STAGE_COMPUTE_BIT, 0,
                       sizeof(AdvancedPushConstants), &push_constants);
  };

  const auto partial_to_reduce_barrier = [&](VkCommandBuffer command_buffer) {
    VkBufferMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.buffer = partial_buffer.buffer;
    barrier.offset = 0;
    barrier.size = partial_buffer.size;
    vkCmdPipelineBarrier(command_buffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 1,
                         &barrier, 0, nullptr);
  };

  const auto record_kernel = [&](VkCommandBuffer command_buffer) {
    if (kernel.info.id == 14) {
      bind_pipeline(command_buffer, *partial_pipeline, *partial_descriptors);
      vkCmdDispatch(command_buffer, ceil_div(options.problem.n, 16),
                    ceil_div(options.problem.m, 16), options.split_count);
      partial_to_reduce_barrier(command_buffer);
      bind_pipeline(command_buffer, *reduce_pipeline, *reduce_descriptors);
      vkCmdDispatch(command_buffer, ceil_div(options.problem.m * options.problem.n, 256),
                    1, 1);
      return;
    }

    if (kernel.info.id == 15) {
      bind_pipeline(command_buffer, *scheduler_pipeline, *scheduler_descriptors);
      vkCmdDispatch(command_buffer, options.scheduler_workgroups, 1, 1);
      return;
    }

    bind_pipeline(command_buffer, *partial_pipeline, *partial_descriptors);
    vkCmdDispatch(command_buffer, options.scheduler_workgroups, 1, 1);
    partial_to_reduce_barrier(command_buffer);
    bind_pipeline(command_buffer, *reduce_pipeline, *reduce_descriptors);
    vkCmdDispatch(command_buffer, ceil_div(options.problem.m * options.problem.n, 256),
                  1, 1);
  };

  for (std::uint32_t iter = 0; iter < options.benchmark.warmup_iterations;
       ++iter) {
    reset_c();
    reset_counter();
    double wall_ms = 0.0;
    (void)record_submit_timed(device, commands, timestamps, false, record_kernel,
                              &wall_ms);
  }

  BenchmarkSamples samples{};
  samples.kernel_ms.reserve(options.benchmark.timed_iterations);
  samples.wall_ms.reserve(options.benchmark.timed_iterations);
  for (std::uint32_t iter = 0; iter < options.benchmark.timed_iterations; ++iter) {
    reset_c();
    reset_counter();
    double wall_ms = 0.0;
    const double kernel_ms = record_submit_timed(
        device, commands, timestamps, options.benchmark.profile, record_kernel,
        &wall_ms);
    samples.kernel_ms.push_back(kernel_ms);
    samples.wall_ms.push_back(wall_ms);
  }

  runtime::download_from_device_buffer(device, commands, c_buffer, c_staging,
                                       bytes_c);
  runtime::copy_from_buffer(c_staging, matrices.c_output.data(), bytes_c);

  RunResult result{};
  result.kernel = kernel.info;
  result.config = kernel.config;
  result.timing.kernel_ms = summarize_samples(samples.kernel_ms);
  result.timing.wall_ms = summarize_samples(samples.wall_ms);
  result.bytes_moved = compute_bytes_moved(options.problem);
  result.gflops =
      compute_gflops(options.problem, result.timing.kernel_ms.average);

  if (options.benchmark.verify) {
    result.verification =
        verify_results(matrices.c_reference, matrices.c_output,
                       options.problem.m, options.problem.n);
  }

  if (!options.benchmark.profile || !timestamps.enabled) {
    result.notes =
        "GPU timestamp profiling unavailable, kernel_ms falls back to wall "
        "time.";
  }

  append_result_csv(options, result);
  return result;
}

RunResult run(const RunOptions& options) {
  validate_problem(options.problem);
  if (options.benchmark.timed_iterations == 0) {
    throw std::runtime_error("timed_iterations must be greater than zero");
  }

  const KernelDefinition& kernel = get_kernel_definition(options.kernel_id);
  if (!kernel.implemented) {
    throw std::runtime_error(
        "Kernel exists in the roadmap but is not implemented yet: " +
        kernel.info.name);
  }

  if (kernel.info.id == 14 || kernel.info.id == 15 || kernel.info.id == 16) {
    return run_advanced_kernel(options, kernel);
  }

  HostMatrices matrices =
      make_host_matrices(options.problem, options.benchmark.random_seed);
  if (options.benchmark.verify) {
    reference_sgemm(options.problem, options.scalars, matrices.a, matrices.b,
                    matrices.c_initial, &matrices.c_reference);
  }

  const std::size_t bytes_a = matrices.a.size() * sizeof(float);
  const std::size_t bytes_b = matrices.b.size() * sizeof(float);
  const std::size_t bytes_c = matrices.c_initial.size() * sizeof(float);

  runtime::Instance instance = runtime::create_instance();
  runtime::DeviceContext device = runtime::create_device(instance.handle);
  runtime::CommandResources commands =
      runtime::create_command_resources(device);

  runtime::Buffer a_buffer =
      runtime::create_device_local_storage_buffer(device, bytes_a);
  runtime::Buffer b_buffer =
      runtime::create_device_local_storage_buffer(device, bytes_b);
  runtime::Buffer c_buffer =
      runtime::create_device_local_storage_buffer(device, bytes_c);
  runtime::Buffer a_staging = runtime::create_staging_buffer(device, bytes_a);
  runtime::Buffer b_staging = runtime::create_staging_buffer(device, bytes_b);
  runtime::Buffer c_staging = runtime::create_staging_buffer(device, bytes_c);

  runtime::copy_to_buffer(a_staging, matrices.a.data(), bytes_a);
  runtime::copy_to_buffer(b_staging, matrices.b.data(), bytes_b);
  runtime::copy_to_buffer(c_staging, matrices.c_initial.data(), bytes_c);
  runtime::upload_to_device_buffer(device, commands, a_staging, a_buffer,
                                   bytes_a);
  runtime::upload_to_device_buffer(device, commands, b_staging, b_buffer,
                                   bytes_b);
  runtime::upload_to_device_buffer(device, commands, c_staging, c_buffer,
                                   bytes_c);

  const KernelConfig effective_config = make_effective_config(kernel, options);

  runtime::DescriptorResources descriptors =
      runtime::create_descriptor_resources(device);
  runtime::update_descriptor_set(device, descriptors, a_buffer, b_buffer,
                                 c_buffer);
  
  std::optional<runtime::AutotuneSpecialization> specialization;

  if (kernel.info.id == 13) {
    specialization = make_specialization(effective_config, options.autotune);
  }

  runtime::ComputePipeline pipeline = runtime::create_compute_pipeline(
      device, descriptors, kernel.shader_path, sizeof(PushConstants),
      specialization ? &*specialization : nullptr);
  runtime::TimestampQuery timestamps = runtime::create_timestamp_query(device);

  const DispatchShape dispatch =
      compute_dispatch_shape(options.problem, effective_config);

  for (std::uint32_t iter = 0; iter < options.benchmark.warmup_iterations;
       ++iter) {
    double wall_ms = 0.0;
    (void)run_dispatch_once(options, device, c_buffer, c_staging,
                            matrices.c_initial, descriptors, pipeline, commands,
                            timestamps, dispatch, false, &wall_ms);
  }

  BenchmarkSamples samples{};
  samples.kernel_ms.reserve(1);
  samples.wall_ms.reserve(1);
  double wall_ms = 0.0;
  const double kernel_ms = run_dispatch_batch(
      options, device, c_buffer, c_staging, matrices.c_initial, descriptors,
      pipeline, commands, timestamps, dispatch,
      options.benchmark.timed_iterations, options.benchmark.profile, &wall_ms);
  samples.kernel_ms.push_back(kernel_ms);
  samples.wall_ms.push_back(wall_ms);

  if (options.benchmark.verify) {
    double verify_wall_ms = 0.0;
    (void)run_dispatch_once(options, device, c_buffer, c_staging,
                            matrices.c_initial, descriptors, pipeline, commands,
                            timestamps, dispatch, false, &verify_wall_ms);
  }

  runtime::download_from_device_buffer(device, commands, c_buffer, c_staging,
                                       bytes_c);
  runtime::copy_from_buffer(c_staging, matrices.c_output.data(), bytes_c);

  RunResult result{};
  result.kernel = kernel.info;
  result.config = effective_config;
  result.timing.kernel_ms = summarize_samples(samples.kernel_ms);
  result.timing.wall_ms = summarize_samples(samples.wall_ms);
  result.bytes_moved = compute_bytes_moved(options.problem);
  result.gflops =
      compute_gflops(options.problem, result.timing.kernel_ms.average);

  if (options.benchmark.verify) {
    result.verification =
        verify_results(matrices.c_reference, matrices.c_output,
                       options.problem.m, options.problem.n);
  }

  if (!options.benchmark.profile || !timestamps.enabled) {
    result.notes =
        "GPU timestamp profiling unavailable, kernel_ms falls back to wall "
        "time.";
  }

  append_result_csv(options, result);
  return result;
}

}  // namespace gemm
