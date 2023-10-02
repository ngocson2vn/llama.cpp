#include <climits>
#include <cstdarg>
#include <fstream>
#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "common.h"
#include "fundamentals.h"
#include "llama.h"

// =======================================================================
// Global state
// =======================================================================
static struct ggml_state g_state;
// static atomic_int g_state_barrier = 0;

static struct ggml_context* ctx_meta = NULL;

// =======================================================================
// Utility functions
// =======================================================================
static void* ggml_aligned_malloc(size_t size) {
  void* aligned_memory = NULL;
  int result = posix_memalign(&aligned_memory, GGML_MEM_ALIGN, size);
  if (result != 0) {
    fprintf(
        stderr,
        "Could not allocate memory chunk with size = %d and alignment = %d\n",
        size, GGML_MEM_ALIGN);
  }

  return aligned_memory;
}

void gguf_free(struct gguf_context* ctx) {
  if (ctx == NULL) {
    return;
  }

  if (ctx->kv) {
    // free string memory - not great..
    for (uint32_t i = 0; i < ctx->header.n_kv; ++i) {
      struct gguf_kv* kv = &ctx->kv[i];

      if (kv->key.data) {
        free(kv->key.data);
      }

      if (kv->type == GGUF_TYPE_STRING) {
        if (kv->value.str.data) {
          free(kv->value.str.data);
        }
      }

      if (kv->type == GGUF_TYPE_ARRAY) {
        if (kv->value.arr.data) {
          if (kv->value.arr.type == GGUF_TYPE_STRING) {
            for (uint32_t j = 0; j < kv->value.arr.n; ++j) {
              struct gguf_str* str = &((struct gguf_str*)kv->value.arr.data)[j];
              if (str->data) {
                free(str->data);
              }
            }
          }
          free(kv->value.arr.data);
        }
      }
    }

    free(ctx->kv);
  }

  if (ctx->infos) {
    for (uint32_t i = 0; i < ctx->header.n_tensors; ++i) {
      struct gguf_tensor_info* info = &ctx->infos[i];

      if (info->name.data) {
        free(info->name.data);
      }
    }

    free(ctx->infos);
  }

  free(ctx);
}

static void ggml_setup_op_has_task_pass(void) {
  {  // INIT
    bool* p = GGML_OP_HAS_INIT;

    p[GGML_OP_ACC] = true;
    p[GGML_OP_MUL_MAT] = true;
    p[GGML_OP_OUT_PROD] = true;
    p[GGML_OP_SET] = true;
    p[GGML_OP_GET_ROWS_BACK] = true;
    p[GGML_OP_DIAG_MASK_INF] = true;
    p[GGML_OP_DIAG_MASK_ZERO] = true;
    p[GGML_OP_CONV_1D] = true;
    p[GGML_OP_CONV_2D] = true;
    p[GGML_OP_CONV_TRANSPOSE_2D] = true;
    p[GGML_OP_FLASH_ATTN_BACK] = true;
    p[GGML_OP_CROSS_ENTROPY_LOSS] = true;
    p[GGML_OP_ADD_REL_POS] = true;
  }

  {  // FINALIZE
    bool* p = GGML_OP_HAS_FINALIZE;

    p[GGML_OP_CROSS_ENTROPY_LOSS] = true;
  }
}

struct ggml_context* ggml_init(struct ggml_init_params params) {
  // make this function thread safe
  // ggml_critical_section_start();

  static bool is_first_call = true;

  if (is_first_call) {
    // initialize time system (required on Windows)
    // ggml_time_init();

    // initialize GELU, Quick GELU, SILU and EXP F32 tables
    {
      // const uint64_t t_start = ggml_time_us(); UNUSED(t_start);

      ggml_fp16_t ii;
      for (int i = 0; i < (1 << 16); ++i) {
        uint16_t ui = i;
        memcpy(&ii, &ui, sizeof(ii));
        const float f = table_f32_f16[i] = GGML_COMPUTE_FP16_TO_FP32(ii);
        table_gelu_f16[i] = GGML_FP32_TO_FP16(ggml_gelu_f32(f));
        table_gelu_quick_f16[i] = GGML_FP32_TO_FP16(ggml_gelu_quick_f32(f));
        table_silu_f16[i] = GGML_FP32_TO_FP16(ggml_silu_f32(f));
        table_exp_f16[i] = GGML_FP32_TO_FP16(expf(f));
      }

      // const uint64_t t_end = ggml_time_us(); UNUSED(t_end);

      // GGML_PRINT_DEBUG("%s: GELU, Quick GELU, SILU and EXP tables initialized
      // in %f ms\n", __func__, (t_end - t_start)/1000.0f);
    }

    // initialize g_state
    {
      // const uint64_t t_start = ggml_time_us(); UNUSED(t_start);

      g_state = (struct ggml_state){
          /*.contexts =*/{{0}},
          /*.numa =*/
          {
              .n_nodes = 0,
              .total_cpus = 0,
          },
      };

      for (int i = 0; i < GGML_MAX_CONTEXTS; ++i) {
        g_state.contexts[i].used = false;
      }

      // const uint64_t t_end = ggml_time_us(); UNUSED(t_end);

      // GGML_PRINT_DEBUG("%s: g_state initialized in %f ms\n", __func__, (t_end
      // - t_start)/1000.0f);
    }

    // #if defined(GGML_USE_CUBLAS)
    //         ggml_init_cublas();
    // #elif defined(GGML_USE_CLBLAST)
    //         ggml_cl_init();
    // #endif

    ggml_setup_op_has_task_pass();

    is_first_call = false;
  }

  // find non-used context in g_state
  struct ggml_context* ctx = NULL;

  for (int i = 0; i < GGML_MAX_CONTEXTS; i++) {
    if (!g_state.contexts[i].used) {
      g_state.contexts[i].used = true;
      ctx = &g_state.contexts[i].context;

      // GGML_PRINT_DEBUG("%s: found unused context %d\n", __func__, i);
      break;
    }
  }

  if (ctx == NULL) {
    // GGML_PRINT_DEBUG("%s: no unused context found\n", __func__);

    // ggml_critical_section_end();

    return NULL;
  }

  // allow to call ggml_init with 0 size
  if (params.mem_size == 0) {
    params.mem_size = GGML_MEM_ALIGN;
  }

  const size_t mem_size = params.mem_buffer
                              ? params.mem_size
                              : GGML_PAD(params.mem_size, GGML_MEM_ALIGN);

  *ctx = (struct ggml_context){
      /*.mem_size           =*/mem_size,
      /*.mem_buffer         =*/params.mem_buffer
          ? params.mem_buffer
          : ggml_aligned_malloc(mem_size),
      /*.mem_buffer_owned   =*/params.mem_buffer ? false : true,
      /*.no_alloc           =*/params.no_alloc,
      /*.no_alloc_save      =*/params.no_alloc,
      /*.n_objects          =*/0,
      /*.objects_begin      =*/NULL,
      /*.objects_end        =*/NULL,
      /*.scratch            =*/
      {
          0,
          0,
          NULL,
      },
      /*.scratch_save       =*/
      {
          0,
          0,
          NULL,
      },
  };

  GGML_ASSERT(ctx->mem_buffer != NULL);

  // ggml_assert_aligned(ctx->mem_buffer);
  GGML_ASSERT(((uintptr_t)(ctx->mem_buffer)) % GGML_MEM_ALIGN == 0);

  // GGML_PRINT_DEBUG("%s: context initialized\n", __func__);

  // ggml_critical_section_end();

  return ctx;
}

size_t ggml_nbytes(const struct ggml_tensor* tensor) {
  size_t nbytes;
  size_t blck_size = type_traits[tensor->type].blck_size;
  if (blck_size == 1) {
    nbytes = type_traits[tensor->type].type_size;
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
      nbytes += (tensor->ne[i] - 1) * tensor->nb[i];
    }
  } else {
    nbytes = tensor->ne[0] * tensor->nb[0] / blck_size;
    for (int i = 1; i < GGML_MAX_DIMS; ++i) {
      nbytes += (tensor->ne[i] - 1) * tensor->nb[i];
    }
  }

  return nbytes;
}

static struct ggml_object* ggml_new_object(struct ggml_context* ctx,
                                           enum ggml_object_type type,
                                           size_t size) {
  // always insert objects at the end of the context's memory pool
  struct ggml_object* obj_cur = ctx->objects_end;

  const size_t cur_offs = obj_cur == NULL ? 0 : obj_cur->offs;
  const size_t cur_size = obj_cur == NULL ? 0 : obj_cur->size;
  const size_t cur_end = cur_offs + cur_size;

  // align to GGML_MEM_ALIGN
  size_t size_needed = GGML_PAD(size, GGML_MEM_ALIGN);

  char* const mem_buffer = (char*)ctx->mem_buffer;
  struct ggml_object* const obj_new =
      (struct ggml_object*)(mem_buffer + cur_end);

  if (cur_end + size_needed + GGML_OBJECT_SIZE > ctx->mem_size) {
    // GGML_PRINT("%s: not enough space in the context's memory pool (needed
    // %zu, available %zu)\n",
    //         __func__, cur_end + size_needed, ctx->mem_size);
    assert(false);
    return NULL;
  }

  *obj_new = (struct ggml_object){
      .offs = cur_end + GGML_OBJECT_SIZE,
      .size = size_needed,
      .next = NULL,
      .type = type,
  };

  // ggml_assert_aligned(mem_buffer + obj_new->offs);
  GGML_ASSERT(((uintptr_t)(mem_buffer + obj_new->offs)) % GGML_MEM_ALIGN == 0);

  if (obj_cur != NULL) {
    obj_cur->next = obj_new;
  } else {
    // this is the first object in this context
    ctx->objects_begin = obj_new;
  }

  ctx->objects_end = obj_new;

  // printf("%s: inserted new object at %zu, size = %zu\n", __func__, cur_end,
  // obj_new->size);

  return obj_new;
}

static struct ggml_tensor* ggml_new_tensor_impl(struct ggml_context* ctx,
                                                enum ggml_type type, int n_dims,
                                                const int64_t* ne,
                                                struct ggml_tensor* view_src,
                                                size_t view_offs) {
  assert(n_dims >= 1 && n_dims <= GGML_MAX_DIMS);

  // find the base tensor and absolute offset
  if (view_src != NULL && view_src->view_src != NULL) {
    view_offs += view_src->view_offs;
    view_src = view_src->view_src;
  }

  size_t data_size =
      type_traits[type].type_size * (ne[0] / type_traits[type].blck_size);
  for (int i = 1; i < n_dims; i++) {
    data_size *= ne[i];
  }

  GGML_ASSERT(view_src == NULL ||
              data_size + view_offs <= ggml_nbytes(view_src));

  void* data = view_src != NULL ? view_src->data : NULL;
  if (data != NULL) {
    data = (char*)data + view_offs;
  }

  size_t obj_alloc_size = 0;

  if (view_src == NULL && !ctx->no_alloc) {
    if (ctx->scratch.data != NULL) {
      // allocate tensor data in the scratch buffer
      if (ctx->scratch.offs + data_size > ctx->scratch.size) {
        // GGML_PRINT("%s: not enough space in the scratch memory pool (needed
        // %zu, available %zu)\n",
        //         __func__, ctx->scratch.offs + data_size, ctx->scratch.size);
        assert(false);
        return NULL;
      }

      data = (char* const)ctx->scratch.data + ctx->scratch.offs;

      ctx->scratch.offs += data_size;
    } else {
      // allocate tensor data in the context's memory pool
      obj_alloc_size = data_size;
    }
  }

  struct ggml_object* const obj_new =
      ggml_new_object(ctx, (ggml_object_type)GGML_OBJECT_TENSOR,
                      GGML_TENSOR_SIZE + obj_alloc_size);

  // TODO: for recoverable errors, we would need to free the data allocated from
  // the scratch buffer here

  struct ggml_tensor* const result =
      (struct ggml_tensor*)((char*)ctx->mem_buffer + obj_new->offs);

  *result = (struct ggml_tensor){
      /*.type         =*/type,
      /*.backend      =*/(ggml_backend)GGML_BACKEND_CPU,
      /*.n_dims       =*/n_dims,
      /*.ne           =*/{1, 1, 1, 1},
      /*.nb           =*/{0, 0, 0, 0},
      /*.op           =*/(ggml_op)GGML_OP_NONE,
      /*.op_params    =*/{0},
      /*.is_param     =*/false,
      /*.grad         =*/NULL,
      /*.src          =*/{NULL},
      /*.perf_runs    =*/0,
      /*.perf_cycles  =*/0,
      /*.perf_time_us =*/0,
      /*.view_src     =*/view_src,
      /*.view_offs    =*/view_offs,
      /*.data         =*/obj_alloc_size > 0 ? (void*)(result + 1) : data,
      /*.name         =*/{0},
      /*.extra        =*/NULL,
      /*.padding      =*/{0},
  };

  // TODO: this should not be needed as long as we don't rely on aligned SIMD
  // loads
  // ggml_assert_aligned(result->data);

  for (int i = 0; i < n_dims; i++) {
    result->ne[i] = ne[i];
  }

  result->nb[0] = type_traits[type].type_size;
  result->nb[1] = result->nb[0] * (result->ne[0] / type_traits[type].blck_size);
  for (int i = 2; i < GGML_MAX_DIMS; i++) {
    result->nb[i] = result->nb[i - 1] * result->ne[i - 1];
  }

  ctx->n_objects++;

  return result;
}

struct ggml_tensor* ggml_new_tensor(struct ggml_context* ctx,
                                    enum ggml_type type, int n_dims,
                                    const int64_t* ne) {
  return ggml_new_tensor_impl(ctx, type, n_dims, ne, NULL, 0);
}

struct ggml_tensor* ggml_set_name(struct ggml_tensor* tensor,
                                  const char* name) {
  strncpy(tensor->name, name, sizeof(tensor->name));
  tensor->name[sizeof(tensor->name) - 1] = '\0';
  return tensor;
}

void ggml_free(struct ggml_context* ctx) {
  // make this function thread safe
  // ggml_critical_section_start();

  bool found = false;

  for (int i = 0; i < GGML_MAX_CONTEXTS; i++) {
    if (&g_state.contexts[i].context == ctx) {
      g_state.contexts[i].used = false;

      GGML_PRINT_DEBUG("%s: context %d has been freed. memory used = %zu\n",
                       __func__, i, ggml_used_mem(ctx));

      if (ctx->mem_buffer_owned) {
        free(ctx->mem_buffer);
      }

      found = true;
      break;
    }
  }

  if (!found) {
    GGML_PRINT_DEBUG("%s: context not found\n", __func__);
  }

  // ggml_critical_section_end();
}

struct ggml_tensor* ggml_get_tensor(struct ggml_context* ctx,
                                    const char* name) {
  struct ggml_object* obj = ctx->objects_begin;

  char* const mem_buffer = (char*)ctx->mem_buffer;

  while (obj != NULL) {
    if (obj->type == GGML_OBJECT_TENSOR) {
      struct ggml_tensor* cur = (struct ggml_tensor*)(mem_buffer + obj->offs);
      if (strcmp(cur->name, name) == 0) {
        return cur;
      }
    }

    obj = obj->next;
  }

  return NULL;
}

static size_t utf8_len(char src) {
  const size_t lookup[] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 3, 4};
  uint8_t highbits = static_cast<uint8_t>(src) >> 4;
  return lookup[highbits];
}

static void replace_all(std::string& s, const std::string& search,
                        const std::string& replace) {
  std::string result;
  for (size_t pos = 0;; pos += search.length()) {
    auto new_pos = s.find(search, pos);
    if (new_pos == std::string::npos) {
      result += s.substr(pos, s.size() - pos);
      break;
    }
    result += s.substr(pos, new_pos - pos) + replace;
    pos = new_pos;
  }
  s = std::move(result);
}

static void zeros(std::ofstream& file, size_t n) {
  char zero = 0;
  for (size_t i = 0; i < n; ++i) {
    file.write(&zero, 1);
  }
}

#ifdef __GNUC__
#ifdef __MINGW32__
#define LLAMA_ATTRIBUTE_FORMAT(...) \
  __attribute__((format(gnu_printf, __VA_ARGS__)))
#else
#define LLAMA_ATTRIBUTE_FORMAT(...) __attribute__((format(printf, __VA_ARGS__)))
#endif
#else
#define LLAMA_ATTRIBUTE_FORMAT(...)
#endif

LLAMA_ATTRIBUTE_FORMAT(1, 2)
static std::string format(const char* fmt, ...) {
  va_list ap;
  va_list ap2;
  va_start(ap, fmt);
  va_copy(ap2, ap);
  int size = vsnprintf(NULL, 0, fmt, ap);
  GGML_ASSERT(size >= 0 && size < INT_MAX);  // NOLINT
  std::vector<char> buf(size + 1);
  int size2 = vsnprintf(buf.data(), size + 1, fmt, ap2);
  GGML_ASSERT(size2 == size);
  va_end(ap2);
  va_end(ap);
  return std::string(buf.data(), size);
}

// =======================================================================
// LLaMa Model
// =======================================================================
#ifdef GGML_USE_CUBLAS
#define llama_host_malloc(n) ggml_cuda_host_malloc(n)
#define llama_host_free(data) ggml_cuda_host_free(data)
#elif GGML_USE_METAL
#define llama_host_malloc(n) ggml_metal_host_malloc(n)
#define llama_host_free(data) ggml_metal_host_free(data)
#elif GGML_USE_CPU_HBM
#define llama_host_malloc(n) hbw_malloc(n)
#define llama_host_free(data) \
  if (data != NULL) hbw_free(data)
#else
#define llama_host_malloc(n) malloc(n)
#define llama_host_free(data) free(data)
#endif

// available llama models
enum e_model {
  MODEL_UNKNOWN,
  MODEL_1B,
  MODEL_3B,
  MODEL_7B,
  MODEL_13B,
  MODEL_15B,
  MODEL_30B,
  MODEL_34B,
  MODEL_40B,
  MODEL_65B,
  MODEL_70B,
};

enum llm_arch {
  LLM_ARCH_LLAMA,
  LLM_ARCH_FALCON,
  LLM_ARCH_BAICHUAN,
  LLM_ARCH_GPT2,
  LLM_ARCH_GPTJ,
  LLM_ARCH_GPTNEOX,
  LLM_ARCH_MPT,
  LLM_ARCH_STARCODER,
  LLM_ARCH_UNKNOWN,
};

struct llama_hparams {
  bool vocab_only;
  uint32_t n_vocab;
  uint32_t n_ctx_train;  // context size the model was trained on
  uint32_t n_embd;
  uint32_t n_head;
  uint32_t n_head_kv;
  uint32_t n_layer;
  uint32_t n_rot;
  uint32_t n_ff;

  float f_norm_eps;
  float f_norm_rms_eps;

  float rope_freq_base_train;
  float rope_freq_scale_train;

  bool operator!=(const llama_hparams& other) const {
    return static_cast<bool>(
        memcmp(this, &other, sizeof(llama_hparams)));  // NOLINT
  }

  uint32_t n_gqa() const { return n_head / n_head_kv; }

  uint32_t n_embd_head() const { return n_embd / n_head; }

  uint32_t n_embd_gqa() const { return n_embd / n_gqa(); }
};

struct llama_vocab {
  using id = int32_t;
  using token = std::string;
  using ttype = llama_token_type;

  struct token_data {
    token text;
    float score;
    ttype type;
  };

  enum llama_vocab_type type = LLAMA_VOCAB_TYPE_SPM;

  std::unordered_map<token, id> token_to_id;
  std::vector<token_data> id_to_token;

  std::map<std::pair<std::string, std::string>, int> bpe_ranks;

  // default LLaMA special tokens
  id special_bos_id = 1;
  id special_eos_id = 2;
  id special_unk_id = 0;
  id special_sep_id = -1;
  id special_pad_id = -1;

  id linefeed_id = 13;

  int find_bpe_rank(std::string token_left, std::string token_right) const {
    replace_all(token_left, " ", "\u0120");
    replace_all(token_left, "\n", "\u010A");
    replace_all(token_right, " ", "\u0120");
    replace_all(token_right, "\n", "\u010A");

    auto it = bpe_ranks.find(std::make_pair(token_left, token_right));
    if (it == bpe_ranks.end()) {
      return -1;
    }

    return it->second;
  }
};

struct llama_layer {
  // normalization
  struct ggml_tensor* attn_norm;
  struct ggml_tensor* attn_norm_b;
  struct ggml_tensor* attn_norm_2;
  struct ggml_tensor* attn_norm_2_b;

  // attention
  struct ggml_tensor* wq;
  struct ggml_tensor* wk;
  struct ggml_tensor* wv;
  struct ggml_tensor* wo;
  struct ggml_tensor* wqkv;

  // attention bias
  struct ggml_tensor* bo;
  struct ggml_tensor* bqkv;

  // normalization
  struct ggml_tensor* ffn_norm;
  struct ggml_tensor* ffn_norm_b;

  // ff
  struct ggml_tensor* w1;  // ffn_gate
  struct ggml_tensor* w2;  // ffn_down
  struct ggml_tensor* w3;  // ffn_up

  // ff bias
  struct ggml_tensor* b2;  // ffn_down
  struct ggml_tensor* b3;  // ffn_up
};

struct llama_buffer {
  void* data = NULL;
  size_t size = 0;

  // fallback to malloc / free
  // useful in cases where CUDA can try to allocate PINNED memory
  bool fallback = false;

  void resize(size_t n) {
    llama_host_free(data);

    data = llama_host_malloc(n);
    if (!data) {
      fallback = true;
      data = malloc(n);
    } else {
      fallback = false;
    }

    GGML_ASSERT(data);
    size = n;
  }

  ~llama_buffer() {
    if (data) {
      if (fallback) {  // NOLINT
        free(data);
      } else {
        llama_host_free(data);
      }
    }

    data = NULL;
  }
};

struct llama_mmap {
  void* addr;
  size_t size;

  llama_mmap(const llama_mmap&) = delete;

#ifdef _POSIX_MAPPED_FILES
  static constexpr bool SUPPORTED = true;

  llama_mmap(struct llama_file* file,
             size_t prefetch = (size_t)-1 /* -1 = max value */,
             bool numa = false) {
    size = file->size;
    int fd = fileno(file->fp);
    int flags = MAP_SHARED;
    // prefetch/readahead impairs performance on NUMA systems
    if (numa) {
      prefetch = 0;
    }
#ifdef __linux__
    if (prefetch) {
      flags |= MAP_POPULATE;
    }
#endif
    addr = mmap(NULL, file->size, PROT_READ, flags, fd, 0);
    if (addr == MAP_FAILED) {
      throw std::runtime_error(format("mmap failed: %s", strerror(errno)));
    }

    if (prefetch > 0) {
      // Advise the kernel to preload the mapped memory
      if (posix_madvise(addr, std::min(file->size, prefetch),
                        POSIX_MADV_WILLNEED)) {
        fprintf(stderr,
                "warning: posix_madvise(.., POSIX_MADV_WILLNEED) failed: %s\n",
                strerror(errno));
      }
    }
    if (numa) {
      // advise the kernel not to use readahead
      // (because the next page might not belong on the same node)
      if (posix_madvise(addr, file->size, POSIX_MADV_RANDOM)) {
        fprintf(stderr,
                "warning: posix_madvise(.., POSIX_MADV_RANDOM) failed: %s\n",
                strerror(errno));
      }
    }
  }

  ~llama_mmap() { munmap(addr, size); }
#elif defined(_WIN32)
  static constexpr bool SUPPORTED = true;

  llama_mmap(struct llama_file* file, bool prefetch = true, bool numa = false) {
    (void)numa;

    size = file->size;

    HANDLE hFile = (HANDLE)_get_osfhandle(_fileno(file->fp));

    HANDLE hMapping =
        CreateFileMappingA(hFile, NULL, PAGE_READONLY, 0, 0, NULL);
    DWORD error = GetLastError();

    if (hMapping == NULL) {
      throw std::runtime_error(format("CreateFileMappingA failed: %s",
                                      llama_format_win_err(error).c_str()));
    }

    addr = MapViewOfFile(hMapping, FILE_MAP_READ, 0, 0, 0);
    error = GetLastError();
    CloseHandle(hMapping);

    if (addr == NULL) {
      throw std::runtime_error(format("MapViewOfFile failed: %s",
                                      llama_format_win_err(error).c_str()));
    }

    if (prefetch) {
      // PrefetchVirtualMemory is only present on Windows 8 and above, so we
      // dynamically load it
      BOOL(WINAPI * pPrefetchVirtualMemory)
      (HANDLE, ULONG_PTR, PWIN32_MEMORY_RANGE_ENTRY, ULONG);
      HMODULE hKernel32 = GetModuleHandleW(L"kernel32.dll");

      // may fail on pre-Windows 8 systems
      pPrefetchVirtualMemory =
          reinterpret_cast<decltype(pPrefetchVirtualMemory)>(
              GetProcAddress(hKernel32, "PrefetchVirtualMemory"));

      if (pPrefetchVirtualMemory) {
        // advise the kernel to preload the mapped memory
        WIN32_MEMORY_RANGE_ENTRY range;
        range.VirtualAddress = addr;
        range.NumberOfBytes = (SIZE_T)size;
        if (!pPrefetchVirtualMemory(GetCurrentProcess(), 1, &range, 0)) {
          fprintf(stderr, "warning: PrefetchVirtualMemory failed: %s\n",
                  llama_format_win_err(GetLastError()).c_str());
        }
      }
    }
  }

  ~llama_mmap() {
    if (!UnmapViewOfFile(addr)) {
      fprintf(stderr, "warning: UnmapViewOfFile failed: %s\n",
              llama_format_win_err(GetLastError()).c_str());
    }
  }
#else
  static constexpr bool SUPPORTED = false;

  llama_mmap(struct llama_file* file, bool prefetch = true, bool numa = false) {
    (void)file;
    (void)prefetch;
    (void)numa;

    throw std::runtime_error(std::string("mmap not supported"));
  }
#endif
};

// Represents some region of memory being locked using mlock or VirtualLock;
// will automatically unlock on destruction.
struct llama_mlock {
  void* addr = NULL;
  size_t size = 0;

  bool failed_already = false;

  llama_mlock() {}
  llama_mlock(const llama_mlock&) = delete;

  ~llama_mlock() {
    if (size) {
      raw_unlock(addr, size);
    }
  }

  void init(void* ptr) {
    GGML_ASSERT(addr == NULL && size == 0);  // NOLINT
    addr = ptr;
  }

  void grow_to(size_t target_size) {
    GGML_ASSERT(addr);
    if (failed_already) {
      return;
    }
    size_t granularity = lock_granularity();
    target_size = (target_size + granularity - 1) & ~(granularity - 1);
    if (target_size > size) {
      if (raw_lock((uint8_t*)addr + size, target_size - size)) {
        size = target_size;
      } else {
        failed_already = true;
      }
    }
  }

#ifdef _POSIX_MEMLOCK_RANGE
  static constexpr bool SUPPORTED = true;

  static size_t lock_granularity() { return (size_t)sysconf(_SC_PAGESIZE); }

#ifdef __APPLE__
#define MLOCK_SUGGESTION                                              \
  "Try increasing the sysctl values 'vm.user_wire_limit' and "        \
  "'vm.global_user_wire_limit' and/or "                               \
  "decreasing 'vm.global_no_user_wire_amount'.  Also try increasing " \
  "RLIMIT_MLOCK (ulimit -l).\n"
#else
#define MLOCK_SUGGESTION "Try increasing RLIMIT_MLOCK ('ulimit -l' as root).\n"
#endif

  bool raw_lock(const void* addr, size_t size) const {
    if (!mlock(addr, size)) {
      return true;
    }

    char* errmsg = std::strerror(errno);
    bool suggest = (errno == ENOMEM);

    // Check if the resource limit is fine after all
    struct rlimit lock_limit;
    if (suggest && getrlimit(RLIMIT_MEMLOCK, &lock_limit)) {
      suggest = false;
    }
    if (suggest && (lock_limit.rlim_max > lock_limit.rlim_cur + size)) {
      suggest = false;
    }

    fprintf(stderr,
            "warning: failed to mlock %zu-byte buffer (after previously "
            "locking %zu bytes): %s\n%s",
            size, this->size, errmsg, suggest ? MLOCK_SUGGESTION : "");
    return false;
  }

#undef MLOCK_SUGGESTION

  static void raw_unlock(void* addr, size_t size) {
    if (munlock(addr, size)) {
      fprintf(stderr, "warning: failed to munlock buffer: %s\n",
              std::strerror(errno));
    }
  }
#elif defined(_WIN32)
  static constexpr bool SUPPORTED = true;

  static size_t lock_granularity() {
    SYSTEM_INFO si;
    GetSystemInfo(&si);
    return (size_t)si.dwPageSize;
  }

  bool raw_lock(void* ptr, size_t len) const {
    for (int tries = 1;; tries++) {
      if (VirtualLock(ptr, len)) {
        return true;
      }
      if (tries == 2) {
        fprintf(stderr,
                "warning: failed to VirtualLock %zu-byte buffer (after "
                "previously locking %zu bytes): %s\n",
                len, size, llama_format_win_err(GetLastError()).c_str());
        return false;
      }

      // It failed but this was only the first try; increase the working
      // set size and try again.
      SIZE_T min_ws_size, max_ws_size;
      if (!GetProcessWorkingSetSize(GetCurrentProcess(), &min_ws_size,
                                    &max_ws_size)) {
        fprintf(stderr, "warning: GetProcessWorkingSetSize failed: %s\n",
                llama_format_win_err(GetLastError()).c_str());
        return false;
      }
      // Per MSDN: "The maximum number of pages that a process can lock
      // is equal to the number of pages in its minimum working set minus
      // a small overhead."
      // Hopefully a megabyte is enough overhead:
      size_t increment = len + 1048576;
      // The minimum must be <= the maximum, so we need to increase both:
      min_ws_size += increment;
      max_ws_size += increment;
      if (!SetProcessWorkingSetSize(GetCurrentProcess(), min_ws_size,
                                    max_ws_size)) {
        fprintf(stderr, "warning: SetProcessWorkingSetSize failed: %s\n",
                llama_format_win_err(GetLastError()).c_str());
        return false;
      }
    }
  }

  static void raw_unlock(void* ptr, size_t len) {
    if (!VirtualUnlock(ptr, len)) {
      fprintf(stderr, "warning: failed to VirtualUnlock buffer: %s\n",
              llama_format_win_err(GetLastError()).c_str());
    }
  }
#else
  static constexpr bool SUPPORTED = false;

  static size_t lock_granularity() { return (size_t)65536; }

  bool raw_lock(const void* addr, size_t len) const {
    fprintf(stderr, "warning: mlock not supported on this system\n");
    return false;
  }

  static void raw_unlock(const void* addr, size_t len) {}
#endif
};

struct llama_model {
  e_model type = MODEL_UNKNOWN;
  llm_arch arch = LLM_ARCH_UNKNOWN;
  llama_ftype ftype = LLAMA_FTYPE_ALL_F32;

  std::string name = "n/a";

  llama_hparams hparams = {};
  llama_vocab vocab;

  struct ggml_tensor* tok_embeddings;
  struct ggml_tensor* pos_embeddings;

  struct ggml_tensor* output_norm;
  struct ggml_tensor* output_norm_b;
  struct ggml_tensor* output;

  std::vector<llama_layer> layers;

  int n_gpu_layers;

  // context
  struct ggml_context* ctx = NULL;

  // the model memory buffer
  llama_buffer buf;

  // model memory mapped file
  std::unique_ptr<llama_mmap> mapping;

  // objects representing data potentially being locked in memory
  llama_mlock mlock_buf;
  llama_mlock mlock_mmap;

  // for quantize-stats only
  std::vector<std::pair<std::string, struct ggml_tensor*>> tensors_by_name;

  int64_t t_load_us = 0;
  int64_t t_start_us = 0;

  ~llama_model() {
    if (ctx) {
      ggml_free(ctx);
    }

#ifdef GGML_USE_CUBLAS
    for (size_t i = 0; i < tensors_by_name.size(); ++i) {
      ggml_cuda_free_data(tensors_by_name[i].second);
    }
    ggml_cuda_free_scratch();
#elif defined(GGML_USE_CLBLAST)
    for (size_t i = 0; i < tensors_by_name.size(); ++i) {
      ggml_cl_free_data(tensors_by_name[i].second);
    }
#endif
  }
};

static void llama_log_callback_default(ggml_log_level level, const char* text,
                                       void* user_data) {
  (void)level;
  (void)user_data;
  fputs(text, stderr);
  fflush(stderr);
}

struct llama_state {
  // We save the log callback globally
  ggml_log_callback log_callback = llama_log_callback_default;
  void* log_callback_user_data = nullptr;
};

static llama_state g_llama_state;

static void llama_log_internal_v(ggml_log_level level, const char* format,
                                 va_list args) {
  va_list args_copy;
  va_copy(args_copy, args);
  char buffer[128];
  int len = vsnprintf(buffer, 128, format, args);
  if (len < 128) {
    g_llama_state.log_callback(level, buffer,
                               g_llama_state.log_callback_user_data);
  } else {
    char* buffer2 = new char[len + 1];
    vsnprintf(buffer2, len + 1, format, args_copy);
    buffer2[len] = 0;
    g_llama_state.log_callback(level, buffer2,
                               g_llama_state.log_callback_user_data);
    delete[] buffer2;
  }
  va_end(args_copy);
}

static void llama_log_internal(ggml_log_level level, const char* format, ...) {
  va_list args;
  va_start(args, format);
  llama_log_internal_v(level, format, args);
  va_end(args);
}

enum llama_fver {
  GGUF_FILE_VERSION_V1 = 1,
  GGUF_FILE_VERSION_V2 = 2,
};

static const char* llama_file_version_name(llama_fver version) {
  switch (version) {
    case GGUF_FILE_VERSION_V1:
      return "GGUF V1 (support until nov 2023)";
    case GGUF_FILE_VERSION_V2:
      return "GGUF V2 (latest)";
  }

  return "unknown";
}

LLAMA_ATTRIBUTE_FORMAT(2, 3)
static void llama_log_internal(ggml_log_level level, const char* format, ...);
static void llama_log_callback_default(ggml_log_level level, const char* text,
                                       void* user_data);

#define LLAMA_LOG_INFO(...) llama_log_internal(GGML_LOG_LEVEL_INFO, __VA_ARGS__)
#define LLAMA_LOG_WARN(...) llama_log_internal(GGML_LOG_LEVEL_WARN, __VA_ARGS__)
#define LLAMA_LOG_ERROR(...) \
  llama_log_internal(GGML_LOG_LEVEL_ERROR, __VA_ARGS__)

// =======================================================================
// Main
// =======================================================================
const char* model_file_path = "models/llama-2-13b-chat.Q4_0.gguf";
FILE* file = nullptr;
uint32_t magic = 0;
size_t offset = 0;
size_t nitems = 0;
size_t ret = 0;
bool ok = true;

llama_model* model = new llama_model();
struct gguf_context* ctx =
    (struct gguf_context*)ggml_aligned_malloc(sizeof(struct gguf_context));

int main() {
  init_type_traits();

  // Open model file
  {
    printf("== Open model file ==\n");
    file = fopen(model_file_path, "rb");
    if (!file) {
      printf("Could not read %s\n", model_file_path);
    }
    printf("OK\n");
    printf("\n");
  }

  // Read magic number
  {
    printf("== Read magic number ==\n");
    nitems = sizeof(magic);
    fread(&magic, 1, nitems, file);
    offset += nitems;
    printf("Magic number = 0x%x\n", magic);
    printf("Current offset = %d; actual file offset = %d\n", offset,
           ftell(file));
    printf("\n");
  }

  // Read header
  {
    printf("== Read header ==\n");
    ctx->header.magic = magic;

    // version
    nitems = sizeof(ctx->header.version);
    fread(&ctx->header.version, 1, nitems, file);
    offset += nitems;
    printf("header.version = %d\n", ctx->header.version);
    printf("Current offset = %d; current file offset = %d\n", offset,
           ftell(file));
    printf("\n");

    // n_tensors
    nitems = sizeof(ctx->header.n_tensors);
    fread(&ctx->header.n_tensors, 1, nitems, file);
    offset += nitems;
    printf("header.n_tensors = %d\n", ctx->header.n_tensors);
    printf("Current offset = %d; current file offset = %d\n", offset,
           ftell(file));
    printf("\n");

    // n_kv
    nitems = sizeof(ctx->header.n_kv);
    fread(&ctx->header.n_kv, 1, nitems, file);
    offset += nitems;
    printf("header.n_kv = %d\n", ctx->header.n_kv);
    printf("Current offset = %d; actual file offset = %d\n", offset,
           ftell(file));
    printf("\n");
  }

  // Read the kv pairs
  {
    printf("== Read the kv pairs ==\n");
    size_t kv_buf_size = ctx->header.n_kv * sizeof(struct gguf_kv);
    void* kv_ptr = malloc(kv_buf_size);
    ctx->kv = (struct gguf_kv*)kv_ptr;

    for (uint32_t i = 0; i < ctx->header.n_kv; ++i) {
      struct gguf_kv* kv = &ctx->kv[i];

      nitems = sizeof(kv->key.n);
      ret = fread(&kv->key.n, 1, nitems, file);
      offset += nitems;
      printf("Key length = %d\n", kv->key.n);

      kv->key.data = (char*)calloc(kv->key.n + 1, 1);
      nitems = kv->key.n;
      ret = fread(kv->key.data, 1, nitems, file);
      offset += nitems;
      printf("Key = %s\n", kv->key.data);

      nitems = sizeof(kv->type);
      fread(&kv->type, 1, nitems, file);
      offset += nitems;
      printf("Type of value = %d\n", kv->type);

      switch (kv->type) {
        case GGUF_TYPE_UINT8: {
          nitems = sizeof(kv->value.uint8);
          fread(&kv->value.uint8, 1, nitems, file);
          offset += nitems;
          printf("Value = %d\n", kv->value.uint8);
          printf("Current offset = %d; actual file offset = %d\n", offset,
                 ftell(file));
          break;
        }
        case GGUF_TYPE_INT8: {
          nitems = sizeof(kv->value.int8);
          fread(&kv->value.int8, 1, nitems, file);
          offset += nitems;
          printf("Value = %d\n", kv->value.int8);
          printf("Current offset = %d; actual file offset = %d\n", offset,
                 ftell(file));
          break;
        }
        case GGUF_TYPE_UINT16: {
          nitems = sizeof(kv->value.uint16);
          fread(&kv->value.uint16, 1, nitems, file);
          offset += nitems;
          printf("Value = %d\n", kv->value.uint16);
          printf("Current offset = %d; actual file offset = %d\n", offset,
                 ftell(file));
          break;
        }
        case GGUF_TYPE_INT16: {
          nitems = sizeof(kv->value.int16);
          fread(&kv->value.int16, 1, nitems, file);
          offset += nitems;
          printf("Value = %d\n", kv->value.int16);
          printf("Current offset = %d; actual file offset = %d\n", offset,
                 ftell(file));
          break;
        }
        case GGUF_TYPE_UINT32: {
          nitems = sizeof(kv->value.uint32);
          fread(&kv->value.uint32, 1, nitems, file);
          offset += nitems;
          printf("Value = %d\n", kv->value.uint32);
          printf("Current offset = %d; actual file offset = %d\n", offset,
                 ftell(file));
          break;
        }
        case GGUF_TYPE_INT32: {
          nitems = sizeof(kv->value.int32);
          fread(&kv->value.int32, 1, nitems, file);
          offset += nitems;
          printf("Value = %d\n", kv->value.int32);
          printf("Current offset = %d; actual file offset = %d\n", offset,
                 ftell(file));
          break;
        }
        case GGUF_TYPE_FLOAT32: {
          nitems = sizeof(kv->value.float32);
          fread(&kv->value.float32, 1, nitems, file);
          offset += nitems;
          printf("Value = %d\n", kv->value.float32);
          printf("Current offset = %d; actual file offset = %d\n", offset,
                 ftell(file));
          break;
        }
        case GGUF_TYPE_UINT64: {
          nitems = sizeof(kv->value.uint64);
          fread(&kv->value.uint64, 1, nitems, file);
          offset += nitems;
          printf("Value = %d\n", kv->value.uint64);
          printf("Current offset = %d; actual file offset = %d\n", offset,
                 ftell(file));
          break;
        }
        case GGUF_TYPE_INT64: {
          nitems = sizeof(kv->value.int64);
          fread(&kv->value.int64, 1, nitems, file);
          offset += nitems;
          printf("Value = %d\n", kv->value.int64);
          printf("Current offset = %d; actual file offset = %d\n", offset,
                 ftell(file));
          break;
        }
        case GGUF_TYPE_FLOAT64: {
          nitems = sizeof(kv->value.float64);
          fread(&kv->value.float64, 1, nitems, file);
          offset += nitems;
          printf("Value = %d\n", kv->value.float64);
          printf("Current offset = %d; actual file offset = %d\n", offset,
                 ftell(file));
          break;
        }
        case GGUF_TYPE_BOOL: {
          nitems = sizeof(kv->value.bool_);
          fread(&kv->value.bool_, 1, nitems, file);
          offset += nitems;
          printf("Value = %d\n", kv->value.bool_);
          printf("Current offset = %d; actual file offset = %d\n", offset,
                 ftell(file));
          break;
        }
        case GGUF_TYPE_STRING: {
          nitems = sizeof(kv->value.str.n);
          fread(&kv->value.str.n, 1, nitems, file);
          offset += nitems;

          kv->value.str.data = (char*)calloc(kv->value.str.n + 1, 1);
          nitems = kv->value.str.n;
          fread(kv->value.str.data, 1, nitems, file);
          offset += nitems;

          printf("Value = %s\n", kv->value.str.data);
          printf("Current offset = %d; actual file offset = %d\n", offset,
                 ftell(file));
          break;
        }
        case GGUF_TYPE_ARRAY: {
          nitems = sizeof(kv->value.arr.type);
          fread(&kv->value.arr.type, 1, nitems, file);
          offset += nitems;
          nitems = sizeof(kv->value.arr.n);
          fread(&kv->value.arr.n, 1, nitems, file);
          offset += nitems;

          switch (kv->value.arr.type) {
            case GGUF_TYPE_UINT8:
            case GGUF_TYPE_INT8:
            case GGUF_TYPE_UINT16:
            case GGUF_TYPE_INT16:
            case GGUF_TYPE_UINT32:
            case GGUF_TYPE_INT32:
            case GGUF_TYPE_FLOAT32:
            case GGUF_TYPE_UINT64:
            case GGUF_TYPE_INT64:
            case GGUF_TYPE_FLOAT64:
            case GGUF_TYPE_BOOL: {
              nitems = kv->value.arr.n * GGUF_TYPE_SIZE[kv->value.arr.type];
              kv->value.arr.data = malloc(nitems);
              fread(kv->value.arr.data, 1, nitems, file);
              offset += nitems;

              printf("Value = %p\n", kv->value.arr.data);
              printf("Current offset = %d; actual file offset = %d\n", offset,
                     ftell(file));

              break;
            }
            case GGUF_TYPE_STRING: {
              size_t buf_size = kv->value.arr.n * sizeof(struct gguf_str);
              kv->value.arr.data = malloc(buf_size);
              struct gguf_str* arr = (struct gguf_str*)kv->value.arr.data;
              for (uint32_t j = 0; j < kv->value.arr.n; ++j) {
                struct gguf_str* p = &arr[j];
                p->n = 0;
                p->data = nullptr;

                nitems = sizeof(p->n);
                fread(&p->n, 1, nitems, file);
                offset += nitems;

                p->data = (char*)calloc(p->n + 1, 1);
                nitems = p->n;
                fread(p->data, 1, nitems, file);
                offset += nitems;
              }

              printf("Value = [");
              size_t i = 0;
              for (; i < kv->value.arr.n; i++) {
                if (i == 0) {
                  printf("\"%s\"", arr[i].data);
                } else if (i == 999) {
                  printf(", ...");
                  break;
                } else {
                  printf(", \"%s\"", arr[i].data);
                }
              }
              if (i < kv->value.arr.n - 1) {
                printf(", \"%s\"", arr[kv->value.arr.n - 1].data);
              }
              printf("]\n");
              printf("Current offset = %d; actual file offset = %d\n", offset,
                     ftell(file));

              break;
            }
            case GGUF_TYPE_ARRAY:
            case GGUF_TYPE_COUNT:
              GGML_ASSERT(false && "invalid type");
              break;
          }

          break;
        }
        case GGUF_TYPE_COUNT:
          GGML_ASSERT(false && "invalid type");
      }

      printf("\n");
      fflush(stdout);
    }
  }

  // Read the tensor infos
  {
    printf("Read the tensor info\n");
    ctx->infos = (struct gguf_tensor_info*)malloc(
        ctx->header.n_tensors * sizeof(struct gguf_tensor_info));
    for (uint32_t i = 0; i < ctx->header.n_tensors; ++i) {
      struct gguf_tensor_info* info = &ctx->infos[i];

      for (int j = 0; j < GGML_MAX_DIMS; ++j) {
        info->ne[j] = 1;
      }

      nitems = sizeof(info->name.n);
      fread(&info->name.n, 1, nitems, file);
      offset += nitems;

      info->name.data = (char*)calloc(info->name.n + 1, 1);
      nitems = info->name.n;
      fread(info->name.data, 1, nitems, file);
      offset += nitems;
      printf("Tensor name = %s\n", info->name.data);

      nitems = sizeof(info->n_dims);
      fread(&info->n_dims, 1, nitems, file);
      offset += nitems;
      printf("Tensor n_dims = %d\n", info->n_dims);

      printf("Tensor dims = [");
      for (uint32_t j = 0; j < info->n_dims; ++j) {
        nitems = sizeof(info->ne[j]);
        fread(&info->ne[j], 1, nitems, file);
        offset += nitems;
        if (j == 0) {
          printf("%d", info->ne[j]);
        } else {
          printf(", %d", info->ne[j]);
        }
      }
      printf("]\n");

      nitems = sizeof(info->type);
      fread(&info->type, 1, nitems, file);
      offset += nitems;
      printf("Tensor data type = %d\n", info->type);

      nitems = sizeof(info->offset);
      fread(&info->offset, 1, nitems, file);
      offset += nitems;
      printf("Tensor offset = %lld\n", info->offset);
      printf("Current offset = %d; actual file offset = %d\n", offset,
             ftell(file));
      printf("\n");
    }
  }

  // Read alignment
  ctx->alignment = GGUF_DEFAULT_ALIGNMENT;
  int alignment_idx = -1;
  const char* alignment_key = "general.alignment";
  for (size_t i = 0; i < ctx->header.n_kv; i++) {
    if (strcmp(alignment_key, ctx->kv[i].key.data) == 0) {
      alignment_idx = i;
    }
  }

  if (alignment_idx != -1) {
    GGML_ASSERT(ctx->kv[alignment_idx].type == GGUF_TYPE_UINT32);
    ctx->alignment = ctx->kv[alignment_idx].value.uint32;
  }

  // We require the data section to be aligned, so take into account any padding
  {
    const size_t offset_pad = offset % ctx->alignment;

    if (offset_pad != 0) {
      offset += ctx->alignment - offset_pad;
      fseek(file, offset, SEEK_SET);
      printf("Current offset = %d; actual file offset = %d\n", offset,
             ftell(file));
    }
  }

  // Store the current file offset - this is where the data section starts
  ctx->offset = offset;

  // Compute the total size of the data section, taking into account the
  // alignment
  {
    ctx->size = 0;
    for (uint32_t i = 0; i < ctx->header.n_tensors; ++i) {
      struct gguf_tensor_info* info = &ctx->infos[i];

      const int64_t element_count = (int64_t)info->ne[0] *
                                    (int64_t)info->ne[1] *
                                    (int64_t)info->ne[2] * (int64_t)info->ne[3];

      if (element_count % type_traits[info->type].blck_size != 0) {
        fprintf(stderr,
                "%s: tensor '%s' number of elements %lld is not a multiple of "
                "block size (%d)\n",
                __func__, info->name.data, element_count,
                type_traits[info->type].blck_size);
        fclose(file);
        gguf_free(ctx);
        return 1;
      }

      const size_t size_cur =
          (element_count * type_traits[info->type].type_size) /
          type_traits[info->type].blck_size;
      ctx->size += GGML_PAD(size_cur, ctx->alignment);
    }
  }

  // Load the tensor data only if requested
  {
    // If the provided gguf_context is no_alloc, then we create "empty" tensors
    // and do not read the binary blob otherwise, we load the binary blob into
    // the created ggml_context as well, and point the "data" members of the
    // ggml_tensor structs to the appropriate locations in the binary blob

    struct gguf_init_params params = {
        .no_alloc = true,
        .ctx = &ctx_meta,
    };

    // compute the exact size needed for the new ggml_context
    const size_t mem_size =
        params.no_alloc
            ? (ctx->header.n_tensors) * (GGML_OBJECT_SIZE + GGML_TENSOR_SIZE)
            : (ctx->header.n_tensors + 1) *
                      (GGML_OBJECT_SIZE + GGML_TENSOR_SIZE) +
                  ctx->size;

    struct ggml_init_params pdata = {
        .mem_size = mem_size,
        .mem_buffer = NULL,
        .no_alloc = params.no_alloc,
    };

    ggml_context* tmp_ctx = ggml_init(pdata);
    *params.ctx = tmp_ctx;

    struct ggml_context* ctx_data = *params.ctx;

    struct ggml_tensor* data = NULL;

    // ggml_set_no_alloc(ctx_data, true);
    ctx_data->no_alloc = true;

    // create the tensors
    for (uint32_t i = 0; i < ctx->header.n_tensors; ++i) {
      const int64_t ne[GGML_MAX_DIMS] = {
          (int64_t)ctx->infos[i].ne[0],
          (int64_t)ctx->infos[i].ne[1],
          (int64_t)ctx->infos[i].ne[2],
          (int64_t)ctx->infos[i].ne[3],
      };

      struct ggml_tensor* cur = ggml_new_tensor(ctx_data, ctx->infos[i].type,
                                                ctx->infos[i].n_dims, ne);

      ok = ok && cur != NULL;

      ggml_set_name(cur, ctx->infos[i].name.data);

      if (!ok) {
        break;
      }

      // point the data member to the appropriate location in the binary blob
      // using the tensor infos
      if (!params.no_alloc) {
        // cur->data = (char *) data->data + ctx->infos[i].offset - ctx->offset;
        // // offset from start of file
        cur->data =
            (char*)data->data + ctx->infos[i].offset;  // offset from data
      }
    }
  }

  // Logging
  int64_t n_elements = 0;
  size_t n_bytes = 0;
  for (int i = 0; i < ctx->header.n_tensors; i++) {
    const char* name = ctx->infos[i].name.data;
    struct ggml_tensor* t = ggml_get_tensor(ctx_meta, name);
    n_elements += (t->ne[0] * t->ne[1] * t->ne[2] * t->ne[3]);
    n_bytes += ggml_nbytes(t);
  }

  LLAMA_LOG_INFO(
      "%s: loaded meta data with %d key-value pairs and %d tensors from %s "
      "(version %s)\n",
      __func__, ctx->header.n_kv, ctx->header.n_tensors, model_file_path,
      llama_file_version_name((llama_fver)ctx->header.version));
}
