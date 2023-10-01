#include <cstdlib>
#include <cstdio>
#include <cstdint>

// =======================================================================
// Constants
// =======================================================================
#define GGML_MEM_ALIGN          16
#define GGML_MAX_DIMS           4
#define GGUF_DEFAULT_ALIGNMENT  32

// =======================================================================
// Macros
// =======================================================================
#define GGML_ASSERT(x) \
    do { \
        if (!(x)) { \
            fprintf(stderr, "GGML_ASSERT: %s:%d: %s\n", __FILE__, __LINE__, #x); \
            abort(); \
        } \
    } while (0)


// =======================================================================
// Data Structures
// =======================================================================
struct gguf_header {
    uint32_t magic;
    uint32_t version;
    uint64_t n_tensors; // GGUFv2
    uint64_t n_kv;      // GGUFv2
};

struct gguf_str {
    uint64_t n;  // GGUFv2
    char* data;
};

enum gguf_type {
    GGUF_TYPE_UINT8   = 0,
    GGUF_TYPE_INT8    = 1,
    GGUF_TYPE_UINT16  = 2,
    GGUF_TYPE_INT16   = 3,
    GGUF_TYPE_UINT32  = 4,
    GGUF_TYPE_INT32   = 5,
    GGUF_TYPE_FLOAT32 = 6,
    GGUF_TYPE_BOOL    = 7,
    GGUF_TYPE_STRING  = 8,
    GGUF_TYPE_ARRAY   = 9,
    GGUF_TYPE_UINT64  = 10,
    GGUF_TYPE_INT64   = 11,
    GGUF_TYPE_FLOAT64 = 12,
    GGUF_TYPE_COUNT,       // marks the end of the enum
};

union gguf_value {
    uint8_t  uint8;
    int8_t   int8;
    uint16_t uint16;
    int16_t  int16;
    uint32_t uint32;
    int32_t  int32;
    float    float32;
    uint64_t uint64;
    int64_t  int64;
    double   float64;
    bool     bool_;

    struct gguf_str str;

    struct {
        enum gguf_type type;

        uint64_t n;  // GGUFv2
        void * data;
    } arr;
};

struct gguf_kv {
    struct gguf_str key;

    enum  gguf_type  type;
    union gguf_value value;
};

struct gguf_context {
    struct gguf_header header;

    struct gguf_kv          * kv;
    struct gguf_tensor_info * infos;

    size_t alignment;
    size_t offset;    // offset of `data` from beginning of file
    size_t size;      // size of `data` in bytes

    //uint8_t * padding;
    void * data;
};

// Designated Initializers
// https://gcc.gnu.org/onlinedocs/gcc-4.1.2/gcc/Designated-Inits.html
static const size_t GGUF_TYPE_SIZE[GGUF_TYPE_COUNT] = {
    [GGUF_TYPE_UINT8]   = sizeof(uint8_t),
    [GGUF_TYPE_INT8]    = sizeof(int8_t),
    [GGUF_TYPE_UINT16]  = sizeof(uint16_t),
    [GGUF_TYPE_INT16]   = sizeof(int16_t),
    [GGUF_TYPE_UINT32]  = sizeof(uint32_t),
    [GGUF_TYPE_INT32]   = sizeof(int32_t),
    [GGUF_TYPE_FLOAT32] = sizeof(float),
    [GGUF_TYPE_BOOL]    = sizeof(bool),
    [GGUF_TYPE_STRING]  = sizeof(struct gguf_str),
    [GGUF_TYPE_ARRAY]   = 0, // undefined
    [GGUF_TYPE_UINT64]  = sizeof(uint64_t),
    [GGUF_TYPE_INT64]   = sizeof(int64_t),
    [GGUF_TYPE_FLOAT64] = sizeof(double),
};

enum ggml_type {
    GGML_TYPE_F32  = 0,
    GGML_TYPE_F16  = 1,
    GGML_TYPE_Q4_0 = 2,
    GGML_TYPE_Q4_1 = 3,
    // GGML_TYPE_Q4_2 = 4, support has been removed
    // GGML_TYPE_Q4_3 (5) support has been removed
    GGML_TYPE_Q5_0 = 6,
    GGML_TYPE_Q5_1 = 7,
    GGML_TYPE_Q8_0 = 8,
    GGML_TYPE_Q8_1 = 9,
    // k-quantizations
    GGML_TYPE_Q2_K = 10,
    GGML_TYPE_Q3_K = 11,
    GGML_TYPE_Q4_K = 12,
    GGML_TYPE_Q5_K = 13,
    GGML_TYPE_Q6_K = 14,
    GGML_TYPE_Q8_K = 15,
    GGML_TYPE_I8,
    GGML_TYPE_I16,
    GGML_TYPE_I32,
    GGML_TYPE_COUNT,
};

struct gguf_tensor_info {
    struct gguf_str name;

    uint32_t n_dims;
    uint64_t ne[GGML_MAX_DIMS];

    enum ggml_type type;

    uint64_t offset; // offset from start of `data`, must be a multiple of `ALIGNMENT`

    // for writing API
    const void* data;
    size_t size;
};

static void* ggml_aligned_malloc(size_t size) {
  void* aligned_memory = NULL;
  int result = posix_memalign(&aligned_memory, GGML_MEM_ALIGN, size);
  if (result != 0) {
    fprintf(stderr, "Could not allocate memory chunk with size = %d and alignment = %d\n", size, GGML_MEM_ALIGN);
  }

  return aligned_memory;
}

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

int main() {
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
    printf("Current offset = %d; actual file offset = %d\n", offset, ftell(file));
    printf("\n");
  }

  struct gguf_context* ctx = (struct gguf_context*)ggml_aligned_malloc(sizeof(struct gguf_context));

  // Read header
  {
    printf("== Read header ==\n");
    ctx->header.magic = magic;

    // version
    nitems = sizeof(ctx->header.version);
    fread(&ctx->header.version, 1, nitems, file);
    offset += nitems;
    printf("header.version = %d\n", ctx->header.version);
    printf("Current offset = %d; current file offset = %d\n", offset, ftell(file));
    printf("\n");

    // n_tensors
    nitems = sizeof(ctx->header.n_tensors);
    fread(&ctx->header.n_tensors, 1, nitems, file);
    offset += nitems;
    printf("header.n_tensors = %d\n", ctx->header.n_tensors);
    printf("Current offset = %d; current file offset = %d\n", offset, ftell(file));
    printf("\n");

    // n_kv
    nitems = sizeof(ctx->header.n_kv);
    fread(&ctx->header.n_kv, 1, nitems, file);
    offset += nitems;
    printf("header.n_kv = %d\n", ctx->header.n_kv);
    printf("Current offset = %d; actual file offset = %d\n", offset, ftell(file));
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
            case GGUF_TYPE_UINT8:   
            {
                nitems = sizeof(kv->value.uint8);
                fread(&kv->value.uint8, 1, nitems, file);
                offset += nitems;
                printf("Value = %d\n", kv->value.uint8);
                printf("Current offset = %d; actual file offset = %d\n", offset, ftell(file));
                break;
            }
            case GGUF_TYPE_INT8:
            {
                nitems = sizeof(kv->value.int8);
                fread(&kv->value.int8, 1, nitems, file);
                offset += nitems;
                printf("Value = %d\n", kv->value.int8);
                printf("Current offset = %d; actual file offset = %d\n", offset, ftell(file));
                break;
            }
            case GGUF_TYPE_UINT16:
            {
                nitems = sizeof(kv->value.uint16);
                fread(&kv->value.uint16, 1, nitems, file);
                offset += nitems;
                printf("Value = %d\n", kv->value.uint16);
                printf("Current offset = %d; actual file offset = %d\n", offset, ftell(file));
                break;
            }
            case GGUF_TYPE_INT16:
            {
                nitems = sizeof(kv->value.int16);
                fread(&kv->value.int16, 1, nitems, file);
                offset += nitems;
                printf("Value = %d\n", kv->value.int16);
                printf("Current offset = %d; actual file offset = %d\n", offset, ftell(file));
                break;
            }
            case GGUF_TYPE_UINT32:
            {
                nitems = sizeof(kv->value.uint32);
                fread(&kv->value.uint32, 1, nitems, file);
                offset += nitems;
                printf("Value = %d\n", kv->value.uint32);
                printf("Current offset = %d; actual file offset = %d\n", offset, ftell(file));
                break;
            }
            case GGUF_TYPE_INT32:
            {
                nitems = sizeof(kv->value.int32);
                fread(&kv->value.int32, 1, nitems, file);
                offset += nitems;
                printf("Value = %d\n", kv->value.int32);
                printf("Current offset = %d; actual file offset = %d\n", offset, ftell(file));
                break;
            }
            case GGUF_TYPE_FLOAT32:
            {
                nitems = sizeof(kv->value.float32);
                fread(&kv->value.float32, 1, nitems, file);
                offset += nitems;
                printf("Value = %d\n", kv->value.float32);
                printf("Current offset = %d; actual file offset = %d\n", offset, ftell(file));
                break;
            }
            case GGUF_TYPE_UINT64:
            {
                nitems = sizeof(kv->value.uint64);
                fread(&kv->value.uint64, 1, nitems, file);
                offset += nitems;
                printf("Value = %d\n", kv->value.uint64);
                printf("Current offset = %d; actual file offset = %d\n", offset, ftell(file));
                break;
            }
            case GGUF_TYPE_INT64:
            {
                nitems = sizeof(kv->value.int64);
                fread(&kv->value.int64, 1, nitems, file);
                offset += nitems;
                printf("Value = %d\n", kv->value.int64);
                printf("Current offset = %d; actual file offset = %d\n", offset, ftell(file));
                break;
            }
            case GGUF_TYPE_FLOAT64:
            {
                nitems = sizeof(kv->value.float64);
                fread(&kv->value.float64, 1, nitems, file);
                offset += nitems;
                printf("Value = %d\n", kv->value.float64);
                printf("Current offset = %d; actual file offset = %d\n", offset, ftell(file));
                break;
            }
            case GGUF_TYPE_BOOL:
            {
                nitems = sizeof(kv->value.bool_);
                fread(&kv->value.bool_, 1, nitems, file);
                offset += nitems;
                printf("Value = %d\n", kv->value.bool_);
                printf("Current offset = %d; actual file offset = %d\n", offset, ftell(file));
                break;
            }
            case GGUF_TYPE_STRING:
            {
                nitems = sizeof(kv->value.str.n);
                fread(&kv->value.str.n, 1, nitems, file);
                offset += nitems;
                
                kv->value.str.data = (char*)calloc(kv->value.str.n + 1, 1);
                nitems = kv->value.str.n;
                fread(kv->value.str.data, 1, nitems, file);
                offset += nitems;

                printf("Value = %s\n", kv->value.str.data);
                printf("Current offset = %d; actual file offset = %d\n", offset, ftell(file));
                break;
            }
            case GGUF_TYPE_ARRAY:
            {
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
                    case GGUF_TYPE_BOOL:
                    {
                        nitems = kv->value.arr.n * GGUF_TYPE_SIZE[kv->value.arr.type];
                        kv->value.arr.data = malloc(nitems);
                        fread(kv->value.arr.data, 1, nitems, file);
                        offset += nitems;

                        printf("Value = %p\n", kv->value.arr.data);
                        printf("Current offset = %d; actual file offset = %d\n", offset, ftell(file));

                        break;
                    }
                    case GGUF_TYPE_STRING:
                    {
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
                        printf("Current offset = %d; actual file offset = %d\n", offset, ftell(file));

                        break;
                    }
                    case GGUF_TYPE_ARRAY:
                    case GGUF_TYPE_COUNT: GGML_ASSERT(false && "invalid type"); break;
                }

                break;
            }
            case GGUF_TYPE_COUNT: GGML_ASSERT(false && "invalid type");
        }

        printf("\n");
        fflush(stdout);
    }
  }

  // Read the tensor infos
  {
    printf("Read the tensor info\n");
    ctx->infos = (struct gguf_tensor_info*)malloc(ctx->header.n_tensors * sizeof(struct gguf_tensor_info));
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
        printf("Current offset = %d; actual file offset = %d\n", offset, ftell(file));
        printf("\n");
    }
  }

  ctx->alignment = GGUF_DEFAULT_ALIGNMENT;
}