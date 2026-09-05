#pragma once
#include <stddef.h>
#ifdef __cplusplus
extern "C" {
#endif

/* Console */
void   kyber_print(const char* msg);
void   kyber_println(const char* msg);
void   kyber_print_int(int v);
void   kyber_print_float(float v);
char*  kyber_input(const char* prompt);     /* caller must free() */

/* File I/O */
int    kyber_file_exists(const char* path);
long   kyber_file_size(const char* path);
char*  kyber_file_read(const char* path);   /* caller must free() */
int    kyber_file_write(const char* path, const char* data);
int    kyber_file_append(const char* path, const char* data);
int    kyber_file_delete(const char* path);

/* Args */
int    kyber_argc(void);
char*  kyber_argv(int i);                   /* do NOT free */

/* Path */
int    kyber_path_exists(const char* path);
void   kyber_path_join(const char* a, const char* b, char* out, size_t out_size);

#ifdef __cplusplus
}
#endif
