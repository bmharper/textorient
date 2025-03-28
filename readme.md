# textorient

textorient is a Go package that runs a small embedded NN (using
[NCNN](https://github.com/Tencent/ncnn)) to determine the orientation of text in
an image. Our NN is trained to produce 1 of 4 outputs:

-   0: 0 degrees
-   1: 90 degrees
-   2: 180 degrees
-   3: 270 degrees

The training program and data for this package can be found at
[textorient-train](https://github.com/bmharper/textorient-train). The neural
network weights and params files `text_angle_classifier.ncnn.bin` and
`text_angle_classifier.param` are included in this package. The model is a
small, efficient model that runs fine on a CPU.

The model's accuracy on the validation set is 80%, which is why we run the model
on approximately 100 randomly sampled tiles from the image, and choose the
majority vote of the predictions.

### How it works

The function `Orient.StraightenAndMakeUpright()` consists of a few steps:

1. Run [docangle](https://github.com/bmharper/docangle) to compute the angle of
   the page (see `docangle` notes below).
2. Rotate the image by the inverse of the angle found in step 1. The image is
   now straight, but it could be rotated by -90, 90, or 180 degrees (-90 is the
   same as 270 degrees).
3. Extract a sample of 200 tiles, each tile 32x32 pixels, from the image. Run
   the neural network on each of these tiles, and get their orientation (0, 90,
   180, or 270 degrees). Pick the majority vote of their orientations, and
   rotate the image by the inverse, so that the page is upright and straight.

About `docangle`:<br> `docangle` runs a dumb algorithm that knows nothing about
text or its orientation, but it is quite good at identifying a few degrees of
rotation of the page. Note that in its default setting, it only looks for slight
rotation (-2.5 to +2.5 degrees). The algorithm is brute force, and would be very
slow if it was run with much more range.

## Example

```go
import (
	"github.com/bmharper/textorient"
	"github.com/bmharper/cimg"
)

func example(img *cimg.Image) error {
	// Load the neural network.
	// You'll typically do this once, because loading has a fixed cost.
	orient, err := textorient.NewOrient()
	if err != nil {
		return err
	}

	param := textorient.NewWhiteLinesParams()
	// Tweak the search range if necessary
	params.MinDeltaDegrees = -2.7
	params.MaxDeltaDegrees = 2.7
	straight, err := orient.StraightenImage(img, params)
	if err != nil {
		return err
	}

	// ... Send straight onto an OCR service, etc.

	return nil
}
```

## NCNN dependency

This package depends on the NCNN library, which we interface with using cgo.

Prebuilt libraries and include files are included in this repo, inside the
`include` and `lib` directories. If your platform is not included, then please
submit a PR.
