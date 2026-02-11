package textorient

// #cgo CPPFLAGS: -I${SRCDIR}/include/ncnn
// #cgo linux,amd64 CPPFLAGS: -fopenmp
// #cgo linux,amd64 LDFLAGS: -L${SRCDIR}/lib -lncnn_linux_amd64 -lgomp
// #cgo darwin,arm64 LDFLAGS: -L${SRCDIR}/lib -lncnn_darwin_arm64
// #include <stdlib.h>
// #include "orient.h"
import "C"

import (
	_ "embed"
	"fmt"
	"math"
	"unsafe"

	"github.com/bmharper/cimg/v2"
	"github.com/bmharper/docangle"
)

//go:embed text_angle_classifier.ncnn.param
var nnParamFile string

//go:embed text_angle_classifier.ncnn.bin
var nnBinFile string

const TileSize = 32

const (
	Angle0   = 0
	Angle90  = 1
	Angle180 = 2
	Angle270 = 3
)

// An Orientation neural network
type Orient struct {
	nn     unsafe.Pointer
	cParam *C.char
	cBin   *C.char
}

// NewOrient creates a new Orient struct and loads the neural network.
// The network must be closed after use, or you will leak C++ memory.
func NewOrient() (*Orient, error) {
	// If we were loading from disk:
	//var baseFilename string
	//// Convert Go string to C string
	//cBaseFilename := C.CString(baseFilename)
	//defer C.free(unsafe.Pointer(cBaseFilename))
	//cOrient := C.LoadOrientationNNFromFiles(cBaseFilename)

	// Loading from memory (files embedded into the Go build):
	// These memory blocks must remain for the duration of the model's life
	cParam := C.CString(nnParamFile)
	cBin := C.CString(nnBinFile)

	cOrient := C.LoadOrientationNNFromMemory(cParam, cBin, C.size_t(len(nnBinFile)))

	if cOrient == nil {
		C.free(unsafe.Pointer(cParam))
		C.free(unsafe.Pointer(cBin))
		return nil, fmt.Errorf("failed to load neural network")
	}
	// Create a new Orient struct
	orient := &Orient{
		nn:     cOrient,
		cParam: cParam,
		cBin:   cBin,
	}
	return orient, nil
}

// Create a new WhiteLinesParams with defaults
func NewWhiteLinesParams() *docangle.WhiteLinesParams {
	return &docangle.WhiteLinesParams{
		MinDeltaDegrees:  -2.5,
		MaxDeltaDegrees:  2.5,
		StepDegrees:      0.1,
		Include90Degrees: true, // The default docangle doesn't include 90 degree rotations, because of their ambiguity (we fix that ambiguity)
		MaxResolution:    1000,
	}
}

// Use github.com/bmharper/docangle to compute the angle of the page, and rotate the image
// to negate that angle. If the angle is 0, return 'img'
func (o *Orient) Straighten(img *cimg.Image, params *docangle.WhiteLinesParams) *cimg.Image {
	gray := img
	if img.Stride != img.Width*img.NChan() || img.Format != cimg.PixelFormatGRAY {
		gray = img.ToGray()
	}
	dimg := docangle.Image{
		Width:  gray.Width,
		Height: gray.Height,
		Pixels: gray.Pixels,
	}
	if params == nil {
		params = NewWhiteLinesParams()
	}
	// We crop the edges of the page. For most documents scanned at small angles (less than 3 degrees),
	// this does not crop any useful context, and it has the benefit of maintaining identical resolution
	// to the original.
	var straightened *cimg.Image
	_, angleDeg := docangle.GetAngleWhiteLines(&dimg, params)
	if angleDeg == 0 {
		return img
	}
	if math.Abs(angleDeg) > 45 {
		straightened = cimg.NewImage(img.Height, img.Width, img.Format)
	} else {
		straightened = cimg.NewImage(img.Width, img.Height, img.Format)
	}
	cimg.Rotate(img, straightened, -angleDeg*math.Pi/180, nil)
	return straightened
}

// Combine Straighten and MakeUpright
func (o *Orient) StraightenAndMakeUpright(img *cimg.Image, params *docangle.WhiteLinesParams) (*cimg.Image, error) {
	return o.MakeUpright(o.Straighten(img, params))
}

// MakeUpright runs the neural network to determine if the page is upright.
// If necessary, rotate the page by -90, 90, or 180 degrees and return the upright image.
// If the page is already upright, return 'img'
func (o *Orient) MakeUpright(img *cimg.Image) (*cimg.Image, error) {
	orient, err := o.GetImageOrientation(img)
	if err != nil {
		return nil, err
	}

	var final *cimg.Image
	if orient == Angle0 {
		final = img
	} else if orient == Angle90 {
		final = cimg.NewImage(img.Height, img.Width, img.Format)
		cimg.Rotate(img, final, -90*math.Pi/180, nil)
	} else if orient == Angle270 {
		final = cimg.NewImage(img.Height, img.Width, img.Format)
		cimg.Rotate(img, final, 90*math.Pi/180, nil)
	} else if orient == Angle180 {
		final = cimg.NewImage(img.Width, img.Height, img.Format)
		cimg.Rotate(img, final, 180*math.Pi/180, nil)
	}

	return final, nil
}

// Run on a whole image, and return one of 4 angles (Angle0, Angle90, Angle180, Angle270)
func (o *Orient) GetImageOrientation(img *cimg.Image) (int, error) {
	tiles := SplitImage(img, 200, TileSize)
	if len(tiles) == 0 {
		return 0, fmt.Errorf("Image is too small")
	}
	angleCount := [4]int{}
	for _, tile := range tiles {
		denseTile := tile
		if !tile.IsDense() {
			// The NN wants a dense image buffer (stride == width).
			// Our SplitImage function returns deep references into the image, which makes the stride of our tiles
			// equal to the tile of the whole image. So we must clone to the tiles to create dense buffers.
			denseTile = tile.Clone()
		}
		orientation, confidence, err := o.getTileOrientation(denseTile.Pixels, denseTile.Width, denseTile.Height)
		if err != nil {
			return 0, err
		}
		if confidence > 0.5 {
			angleCount[orientation]++
		}
	}
	// Find the most common angle
	maxCount := 0
	bestOrient := 0
	for i := 0; i < 4; i++ {
		if angleCount[i] > maxCount {
			maxCount = angleCount[i]
			bestOrient = i
		}
	}
	return bestOrient, nil
}

// Run the orientation network on a single 32x32 grayscale (8-bit) image.
// orientation returned is one of 0,1,2,3 for 0,90,180,270 degrees.
// confidence is a value between 0 and 1 indicating the confidence of the prediction.
func (o *Orient) getTileOrientation(image []byte, width, height int) (orientation int, confidence float32, err error) {
	if width != TileSize || height != TileSize {
		return 0, 0, fmt.Errorf("image must be 32x32 pixels")
	}
	if o.nn == nil {
		return 0, 0, fmt.Errorf("neural network is not loaded")
	}
	output := [4]float32{}
	result := C.RunOrientationNN(o.nn, unsafe.Pointer(&image[0]), C.int(width), C.int(height), (*C.float)(&output[0]))
	if result != 0 {
		return 0, 0, fmt.Errorf("failed to run neural network")
	}
	vmax := float32(0)
	vmax2 := float32(0)
	for i := 0; i < 4; i++ {
		if output[i] > vmax {
			vmax2 = vmax
			vmax = output[i]
			orientation = i
		}
	}
	confidence = vmax - vmax2
	return orientation, confidence, nil
}

// Close the neural network (free the C++ NCNN object)
func (o *Orient) Close() {
	C.free(unsafe.Pointer(o.cParam))
	C.free(unsafe.Pointer(o.cBin))
	if o.nn != nil {
		C.FreeOrientationNN(o.nn)
		o.nn = nil
	}
}
