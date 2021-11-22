#include "experimental_pmem_allocator.h"

#include <string.h>
#include <unistd.h>

#include <mutex>
#include <thread>

#include "libpmem.h"

namespace tensorflow {

std::atomic<uint64_t> ExperimentalPMemAllocator::next_instance_(0);
thread_local std::vector<AllocatorThread>
    ExperimentalPMemAllocator::access_threads_(0);

ExperimentalPMemAllocator*
ExperimentalPMemAllocator::NewExperimentalPMemAllocator(
    const std::string& pmem_file, uint64_t pmem_size,
    uint32_t max_access_threads, bool devdax_mode,
    const ExperimentalPMemAllocatorConfig& config) {
  int cnt = 0;
  if (++cnt > 1) {
    LOG(WARNING) << "###### create pmem allocator" << pmem_file << " ########";
  }
  if (!ExperimentalPMemAllocator::ValidateConfig(config)) {
    return nullptr;
  }
  int is_pmem;
  uint64_t mapped_size;
  char* pmem;
  if (!devdax_mode) {
    if ((pmem = (char*)pmem_map_file(pmem_file.c_str(), pmem_size,
                                     PMEM_FILE_CREATE, 0666, &mapped_size,
                                     &is_pmem)) == nullptr) {
      fprintf(stderr, "PMem map file %s failed: %s\n", pmem_file.c_str(),
              strerror(errno));
      return nullptr;
    }

    if (!is_pmem) {
      fprintf(stderr, "%s is not a pmem path\n", pmem_file.c_str());
      return nullptr;
    }

  } else {
    if (!CheckDevDaxAndGetSize(pmem_file.c_str(), &mapped_size)) {
      fprintf(stderr, "CheckDevDaxAndGetSize %s failed device %s faild: %s\n",
              pmem_file.c_str(), strerror(errno));
      return nullptr;
    }

    int flags = PROT_READ | PROT_WRITE;
    int fd = open(pmem_file.c_str(), O_RDWR, 0666);
    if (fd < 0) {
      fprintf(stderr, "Open devdax device %s faild: %s\n", pmem_file.c_str(),
              strerror(errno));
      return nullptr;
    }

    if ((pmem = (char*)mmap(nullptr, pmem_size, flags, MAP_SHARED, fd, 0)) ==
        nullptr) {
      fprintf(stderr, "Mmap devdax device %s faild: %s\n", pmem_file.c_str(),
              strerror(errno));
      return nullptr;
    }
  }

  if (mapped_size != pmem_size) {
    fprintf(stderr, "Pmem map file %s size %lu is not same as expected %lu\n",
            pmem_file.c_str(), mapped_size, pmem_size);
    return nullptr;
  }

  ExperimentalPMemAllocator* allocator = nullptr;
  try {
    allocator = new ExperimentalPMemAllocator(pmem, pmem_size,
                                              max_access_threads, config);
  } catch (std::bad_alloc& err) {
    fprintf(stderr, "Error while initialize ExperimentalPMemAllocator: %s\n",
            err.what());
    return nullptr;
  }
  printf("Map pmem space done\n");
  return allocator;
}

void ExperimentalPMemAllocator::SpaceEntryPool::MoveEntryList(
    std::vector<void*>& src, uint32_t b_size) {
  std::lock_guard<SpinMutex> lg(spins_[b_size]);
  assert(b_size < pool_.size());
  pool_[b_size].emplace_back();
  pool_[b_size].back().swap(src);
}

bool ExperimentalPMemAllocator::SpaceEntryPool::FetchEntryList(
    std::vector<void*>& dst, uint32_t b_size) {
  std::lock_guard<SpinMutex> lg(spins_[b_size]);
  if (pool_[b_size].size() != 0) {
    dst.swap(pool_[b_size].back());
    pool_[b_size].pop_back();
    return true;
  }
  return false;
}

void ExperimentalPMemAllocator::BackgroundWork() {
  while (1) {
    if (closing_) return;
    usleep(bg_thread_interval_ * 1000000);
    // Move cached list to pool
    std::vector<void*> moving_list;
    for (auto& tc : thread_cache_) {
      moving_list.clear();
      for (size_t b_size = 1; b_size < tc.freelists.size(); b_size++) {
        moving_list.clear();
        std::unique_lock<SpinMutex> ul(tc.locks[b_size]);

        if (tc.freelists[b_size].size() >= kMinMovableListSize) {
          if (tc.freelists[b_size].size() >= kMinMovableListSize) {
            moving_list.swap(tc.freelists[b_size]);
          }
        }
        if (moving_list.size() > 0) {
          pool_.MoveEntryList(moving_list, b_size);
        }
      }
    }
  }
}

ExperimentalPMemAllocator::ExperimentalPMemAllocator(
    char* pmem, uint64_t pmem_size, uint32_t max_access_threads,
    const ExperimentalPMemAllocatorConfig& config)
    : pmem_(pmem),
      pmem_size_(pmem_size),
      segment_size_(config.segment_size),
      block_size_(config.allocation_unit),
      max_classified_record_block_size_(
          CalculateBlockSize(config.max_allocation_size)),
      bg_thread_interval_(config.bg_thread_interval),
      max_allocation_size_(config.max_allocation_size),
      pool_(max_classified_record_block_size_),
      segment_head_(0),
      segment_record_size_(pmem_size / segment_size_, 0),
      thread_cache_(max_access_threads, max_classified_record_block_size_),
      thread_manager_(std::make_shared<ThreadManager>(max_access_threads)),
      closing_(false),
      instance_id_(next_instance_.fetch_add(1, std::memory_order_relaxed)) {
  if (instance_id_ > next_instance_) {
    fprintf(stderr, "too many instance created (>%lu), abort\n", kMaxInstance);
    throw std::bad_alloc();
  }
  init_data_size_2_block_size();
  if (bg_thread_interval_ > 0) {
    bg_threads_.emplace_back(&ExperimentalPMemAllocator::BackgroundWork, this);
  }
}

void ExperimentalPMemAllocator::DeallocateRaw(void* addr) {
  if (addr == nullptr) {
    return;
  }

  int t_id = MaybeInitAccessThread();

  if (t_id < 0) {
    fprintf(stderr, "too many thread access allocator!\n");
    std::abort();
  }

  uint64_t segment = Addr2Segment(addr);
  if (segment == kPMemNull) return;
  uint32_t b_size = segment_record_size_[segment];
  assert(b_size > 0);

  if (b_size > 0) {
    auto& thread_cache = thread_cache_[t_id];
    // Conflict with bg thread happens only if free entries more than
    // kMinMovableListSize
    std::unique_lock<SpinMutex> ul(thread_cache.locks[b_size]);
    assert(b_size < thread_cache.freelists.size());
    thread_cache.freelists[b_size].emplace_back(addr);
  }
}

void ExperimentalPMemAllocator::PopulateSpace() {
  printf("Polulating PMem space ...\n");
  std::vector<std::thread> ths;

  int pu = 16;  // 16 is a moderate concurrent number for writing PMem.
  for (int i = 0; i < pu; i++) {
    ths.emplace_back([=]() {
      uint64_t offset = pmem_size_ * i / pu;
      // To cover the case that mapped_size_ is not divisible by pu.
      uint64_t len = std::min(pmem_size_ / pu, pmem_size_ - offset);
      // pmem_memset(pmem_ + offset, 0, len, PMEM_F_MEM_NONTEMPORAL);
      memset(pmem_ + offset, 0, len);
    });
  }
  for (auto& t : ths) {
    t.join();
  }
  printf("Populating done\n");
}

ExperimentalPMemAllocator::~ExperimentalPMemAllocator() {
  closing_ = true;
  for (auto& t : bg_threads_) {
    t.join();
  }
  pmem_unmap(pmem_, pmem_size_);
}

bool ExperimentalPMemAllocator::AllocateSegmentSpace(Segment* segment,
                                                     uint32_t record_size) {
  while (1) {
    uint64_t new_segment = segment_head_.load(std::memory_order_relaxed);
    if (new_segment * segment_size_ + segment_size_ < pmem_size_) {
      if (segment_head_.compare_exchange_strong(new_segment, new_segment + 1)) {
        *segment = Segment{Segment2Addr(new_segment), segment_size_};
        segment_record_size_[new_segment] = record_size;
        return true;
      }
      continue;
    }
    return false;
  }
}

void* ExperimentalPMemAllocator::AllocateRaw(size_t alignment, size_t size) {
  void* ret = nullptr;
  int t_id = MaybeInitAccessThread();
  if (t_id < 0) {
    fprintf(stderr, "too many thread access allocator!\n");
    return nullptr;
  }
  uint32_t b_size = Size2BlockSize(size);
  uint32_t aligned_size = b_size * block_size_;
  if (aligned_size > max_allocation_size_ || aligned_size == 0) {
    fprintf(
        stderr,
        "allocating size: %lu size, size is 0 or larger than PMem allocator "
        "max allocation size %lu\n",
        size, max_allocation_size_);
    return nullptr;
  }
  auto& thread_cache = thread_cache_[t_id];
  for (auto i = b_size; i < thread_cache.freelists.size(); i++) {
    if (thread_cache.segments[i].size < aligned_size) {
      // Fetch free list from pool
      {
        std::unique_lock<SpinMutex> ul(thread_cache.locks[i]);
        if (thread_cache.freelists[i].empty()) {
          pool_.FetchEntryList(thread_cache.freelists[i], i);
        }
        // Get space from free list
        if (thread_cache.freelists[i].size() > 0) {
          ret = thread_cache.freelists[i].back();
          thread_cache.freelists[i].pop_back();
          break;
        }
      }
      // Allocate a new segment for requesting block size
      if (!AllocateSegmentSpace(&thread_cache.segments[b_size], b_size)) {
        continue;
      } else {
        i = b_size;
      }
    }
    assert(thread_cache.segments[i].size >= aligned_size);
    ret = thread_cache.segments[i].addr;
    thread_cache.segments[i].size -= aligned_size;
    thread_cache.segments[i].addr =
        (char*)thread_cache.segments[i].addr + aligned_size;
    break;
  }
  return ret;
}

REGISTER_MEM_ALLOCATOR("ExperimentalPMEMAllocator", 30,
                       ExperimentalPMEMAllocatorFactory);
}  // namespace tensorflow
