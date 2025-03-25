#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <string>

// NCNN includes
#include "net.h"
#include "datareader.h"

extern "C" {

// Load a model file. filename is the base name, without the '.ncnn.param' and '.ncnn.bin' extensions.
// Returns null if the model fails to load.
void* LoadOrientationNNFromFiles(const char* filename) {
	ncnn::Net* net = new ncnn::Net();

	std::string fnParam = filename;
	std::string fnBin   = filename;
	fnParam += ".ncnn.param";
	fnBin += ".ncnn.bin";

	// Load model
	if (net->load_param(fnParam.c_str()) != 0) {
		fprintf(stderr, "Error: Failed to load param file '%s'\n", fnParam.c_str());
		return nullptr;
	}
	if (net->load_model(fnBin.c_str()) != 0) {
		fprintf(stderr, "Error: Failed to load model file '%s'\n", fnBin.c_str());
		return nullptr;
	}

	return net;
}

// Load from memory.
// The memory must remain after this function call, and may only be freed after deleting the model.
void* LoadOrientationNNFromMemory(const char* param, const char* bin, size_t binBytes) {
	ncnn::Net* net = new ncnn::Net();

	// Load model from memory
	int paramR = net->load_param_mem((const char*) param);
	if (paramR != 0) {
		fprintf(stderr, "Error: Failed to load param from memory (error %d)\n", paramR);
		return nullptr;
	}

	int binBytesLoaded = net->load_model((const unsigned char*) bin);
	if (binBytesLoaded != binBytes) {
		fprintf(stderr, "Error: Failed to load model bin from memory (%d bytes read instead of %d)\n", (int) binBytesLoaded, (int) binBytes);
		return nullptr;
	}

	return net;
}

// image is a 32x32 grayscale 8-bit image
// output contains the 4 orientation probabilities (0,90,180,270), softmaxed
// Returns 0 on success
int RunOrientationNN(const void* _nn, const void* image, int width, int height, float* output) {
	ncnn::Net* nn = (ncnn::Net*) _nn;

	if (width != 32 || height != 32) {
		fprintf(stderr, "Error: Image must be 32x32 pixels, got %dx%d\n", width, height);
		return -1;
	}

	// Prepare input
	ncnn::Mat in = ncnn::Mat::from_pixels((const unsigned char*) image, ncnn::Mat::PIXEL_GRAY, 32, 32);

	// Convert to float and normalize
	float norm[1] = {1.0f / 255.0f};
	in.substract_mean_normalize(0, norm); // Normalize 0-255 to 0-1

	// Run inference
	ncnn::Extractor ex = nn->create_extractor();
	ex.input("in0", in);
	//assert(r == 0);

	ncnn::Mat out;
	ex.extract("out0", out);
	//assert(r == 0);

	// Verify output dimensions
	if (out.c != 1 || out.w != 4) {
		fprintf(stderr, "Error: Unexpected output shape: %dx%dx%d\n", out.w, out.h, out.c);
		return -1;
	}

	// clamp and softmax
	float sum = 0;
	for (int i = 0; i < 4; i++) {
		float v = out[i];
		if (isnan(v) || isinf(v)) {
			v = -100;
		}
		v         = exp(v);
		output[i] = v;
		sum += v;
	}
	for (int i = 0; i < 4; i++) {
		output[i] = output[i] / sum;
	}

	return 0;
}

void FreeOrientationNN(void* _nn) {
	ncnn::Net* nn = (ncnn::Net*) _nn;
	delete nn;
}
}