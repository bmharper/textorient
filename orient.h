#include <stdlib.h>
#ifdef __linux__
#include <malloc.h>
#endif

void* LoadOrientationNNFromFiles(const char* filename);
void* LoadOrientationNNFromMemory(const char* param, const char* bin, size_t binBytes);
int   RunOrientationNN(const void* nn, const void* image, int width, int height, float* output);
void  FreeOrientationNN(void* nn);
