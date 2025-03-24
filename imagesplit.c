#include "imagesplit.h"

float horizontal_perplexity(const byte* pixels, int width, int height, int stride) {
	int perplexity = 0;

	// Process each row independently
	for (int y = 0; y < height; y++) {
		const byte* row = pixels + y * stride;

		// Initialize the first 5-pixel window
		int window_sum = 0;
		for (int x = 0; x < 5 && x < width; x++) {
			window_sum += row[x];
		}

		// If width < 5, no point in computing perplexity
		if (width < 5) {
			continue;
		}

		int prev_avg = window_sum;

		// Slide the window across the row
		for (int x = 5; x < width; x++) {
			// Update running sum by removing leftmost pixel and adding new pixel
			window_sum   = window_sum - row[x - 5] + row[x];
			int curr_avg = window_sum;

			// Compare current pixel with previous window average
			int diff = row[x] * 5 - prev_avg;
			if (diff < 0)
				diff = -diff; // Absolute difference

			if (diff > 50) {
				perplexity++;
			}

			prev_avg = curr_avg;
		}
	}

	return (float) perplexity / (width * height);
}