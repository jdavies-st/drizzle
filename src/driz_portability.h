#ifdef _WIN32
#define inline_macro        __inline
#define always_inline_macro __forceinline
#else
/*
 * assume gcc for now
 */
#define inline_macro        inline
#define always_inline_macro inline __attribute__((always_inline))
#endif

#define private
