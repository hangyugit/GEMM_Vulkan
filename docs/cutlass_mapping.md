# CUTLASS to Vulkan Mapping

- threadblock tile -> workgroup tile
- warp tile -> subgroup tile
- global/shared iterators -> storage-buffer/shared-memory load paths
- mainloop pipeline -> double-buffered K-tile loop
- epilogue -> final accumulator writeback path
