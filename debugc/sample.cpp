#include <stdio.h>
#include <vector>
#include <assert.h>

// This plain C++ file is kept around for debugging NCNN issues via gdb.
// If we're running via cgo, then invoking gdb is very painful, so we keep this
// plain C++ program around for each of debugging.

// To build this, you'll need to download stb_image.h and place it in the same directory as this file.

// Debug
// mkdir ncnn/debug && cd ncnn/debug
// cmake -DCMAKE_BUILD_TYPE=Debug -DNCNN_BUILD_TOOLS=ON -DNCNN_BUILD_EXAMPLES=OFF -DNCNN_BUILD_BENCHMARK=OFF -DNCNN_BUILD_TESTS=OFF ..
// make -jX
// g++ -g -O0 -o sample debugc/sample.cpp -lncnnd -lgomp -I./ncnn/debug/src -I./ncnn/src -I./debugc -L./ncnn/debug/src

// Release
// mkdir ncnn/build && cd ncnn/build
// cmake -DNCNN_BUILD_TOOLS=ON -DNCNN_BUILD_EXAMPLES=OFF -DNCNN_BUILD_BENCHMARK=OFF -DNCNN_BUILD_TESTS=OFF ..
// make -jX
// g++ -O3 -o sample debugc/sample.cpp -lncnn -lgomp -I./ncnn/build/src -I./ncnn/src -I./debugc -L./ncnn/build/src

// NCNN
#include "net.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

bool LoadFileIntoMemory(const char* filename, void*& buf, size_t& size) {
	FILE* fp = fopen(filename, "rb");
	if (!fp) {
		fprintf(stderr, "Error: Failed to open file '%s'\n", filename);
		return false;
	}
	fseek(fp, 0, SEEK_END);
	size = ftell(fp);
	fseek(fp, 0, SEEK_SET);
	buf = malloc(size + 1);
	if (!buf) {
		fprintf(stderr, "Error: Failed to allocate memory for file '%s'\n", filename);
		fclose(fp);
		return false;
	}
	size_t bytes_read = fread(buf, 1, size, fp);
	if (bytes_read != size) {
		fprintf(stderr, "Error: Failed to read file '%s'\n", filename);
		free(buf);
		fclose(fp);
		return false;
	}
	((char*) buf)[size] = 0; // Null-terminate the buffer
	fclose(fp);
	return true;
}

int main(int argc, char** argv) {
	if (argc != 2) {
		fprintf(stderr, "Usage: %s [image_path]\n", argv[0]);
		return -1;
	}

	const char* image_path = argv[1];

	// Load image using stb_image
	int            width, height, channels;
	unsigned char* img_data = stbi_load(image_path, &width, &height, &channels, 1); // Force 1 channel (grayscale)

	if (!img_data) {
		fprintf(stderr, "Error: Could not load image %s\n", image_path);
		return -1;
	}

	if (width != 32 || height != 32) {
		fprintf(stderr, "Error: Image must be 32x32 pixels, got %dx%d\n", width, height);
		stbi_image_free(img_data);
		return -1;
	}

	// Initialize NCNN
	ncnn::Net net;

	void*  mModel = nullptr;
	void*  mParam = nullptr;
	size_t sModel = 0;
	size_t sParam = 0;

	LoadFileIntoMemory("text_angle_classifier.ncnn.bin", mModel, sModel);
	LoadFileIntoMemory("text_angle_classifier.ncnn.param", mParam, sParam);

	if (net.load_param_mem((const char*) mParam) != 0) {
		fprintf(stderr, "Error: Failed to load param file from memory\n");
		stbi_image_free(img_data);
		return -1;
	}

	int rr = net.load_model((const unsigned char*) mModel);
	if (rr != sModel) {
		fprintf(stderr, "Error: Failed to load model weights file from memory\n");
		stbi_image_free(img_data);
		return -1;
	}

	// Load model
	//	if (net.load_param("text_angle_classifier.ncnn.param") != 0) {
	//		fprintf(stderr, "Error: Failed to load param file\n");
	//		stbi_image_free(img_data);
	//		return -1;
	//	}
	//
	//	if (net.load_model("text_angle_classifier.ncnn.bin") != 0) {
	//		fprintf(stderr, "Error: Failed to load model file\n");
	//		stbi_image_free(img_data);
	//		return -1;
	//	}

	// Prepare input
	ncnn::Mat in = ncnn::Mat::from_pixels(img_data, ncnn::Mat::PIXEL_GRAY, 32, 32);

	// Convert to float and normalize
	//in            = in.clone();
	float norm[1] = {1.0f / 255.0f};
	in.substract_mean_normalize(0, norm); // Normalize 0-255 to 0-1

	// Run inference
	ncnn::Extractor ex = net.create_extractor();
	int             r  = ex.input("in0", in);
	assert(r == 0);

	ncnn::Mat out;
	r = ex.extract("out0", out);
	assert(r == 0);

	// Verify output dimensions
	if (out.c != 1 || out.w != 4) {
		fprintf(stderr, "Error: Unexpected output shape: %dx%dx%d\n", out.w, out.h, out.c);
		stbi_image_free(img_data);
		return -1;
	}

	// softmax
	float sum = 0;
	for (int i = 0; i < 4; i++) {
		sum += exp(out[i]);
	}
	for (int i = 0; i < 4; i++) {
		out[i] = exp(out[i]) / sum;
	}

	// Print results
	printf("Output vector: [");
	//auto row = out.row(0);
	for (int i = 0; i < 4; i++) {
		printf("%.6f", out[i]);
		if (i < 3)
			printf(", ");
	}
	printf("]\n");

	// Cleanup
	stbi_image_free(img_data);

	return 0;
}