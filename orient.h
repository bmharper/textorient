#include <malloc.h>

void* LoadOrientationNN(const char* filename);
int   RunOrientationNN(const void* nn, const void* image, int width, int height, float* output);
void  FreeOrientationNN(void* nn);
