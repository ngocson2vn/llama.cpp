#include <cstdlib>
#include <cstdio>
#include <cstdint>

// =======================================================================
// Define
// =======================================================================
#define GGML_MEM_ALIGN 16


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
    char * data;
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

int main() {
  // Open model file
  {
    printf("Open model file\n");
    file = fopen(model_file_path, "rb");
    if (!file) {
      printf("Could not read %s\n", model_file_path);
    }
    printf("OK\n");
    printf("\n");
  }

  // Read magic number
  {
    printf("Read magic number\n");
    nitems = sizeof(magic);
    fread(&magic, 1, nitems, file);
    offset += nitems;
    printf("Magic number = 0x%x\n", magic);
    printf("Current offset = %d; actual file offset = %d\n", offset, ftell(file));
    printf("\n");
  }

  // Read header
  {
    printf("Read header\n");
    struct gguf_context* ctx = (struct gguf_context*)ggml_aligned_malloc(sizeof(struct gguf_context));
    ctx->header.magic = magic;

    // n_tensors
    nitems = sizeof(ctx->header.n_tensors);
    fread(&ctx->header.n_tensors, 1, nitems, file);
    offset += nitems;
    printf("Current offset = %d; current file offset = %d\n", offset, ftell(file));

    // n_kv
    nitems = sizeof(ctx->header.n_kv);
    fread(&ctx->header.n_kv, 1, nitems, file);
    offset += nitems;
    printf("Current offset = %d; current file offset = %d\n", offset, ftell(file));
    printf("\n");
  }

  // Read the kv pairs
  {

  }
}