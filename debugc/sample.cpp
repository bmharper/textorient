#include <stdio.h>
#include <vector>
#include <assert.h>

// Debug
// g++ -g -O0 -o sample debugc/sample.cpp -lncnn -lgomp -I./ncnn/build/src -I./ncnn/src -I./debugc -L./ncnn/build/src

// Release
// g++ -O3 -o sample debugc/sample.cpp -lncnn -lgomp -I./ncnn/build/src -I./ncnn/src -I./debugc -L./ncnn/build/src

// NCNN includes
#include "net.h"

// STB image includes
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

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

	// Load model
	if (net.load_param("text_angle_classifier.ncnn.param") != 0) {
		fprintf(stderr, "Error: Failed to load param file\n");
		stbi_image_free(img_data);
		return -1;
	}

	if (net.load_model("text_angle_classifier.ncnn.bin") != 0) {
		fprintf(stderr, "Error: Failed to load model file\n");
		stbi_image_free(img_data);
		return -1;
	}

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