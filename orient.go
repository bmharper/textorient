package textorient

// #cgo CPPFLAGS: -fopenmp -I${SRCDIR}/ncnn/build/src -I${SRCDIR}/ncnn/src
// #cgo LDFLAGS: -L${SRCDIR}/ncnn/build/src -lncnn -lgomp
// #include <stdlib.h>
// #include "orient.h"
import "C"
import (
	"fmt"
	"unsafe"
)

// An Orientation neural network
type Orient struct {
	nn unsafe.Pointer
}

// LoadNN loads a neural network from a file.
// The network must be closed after use, or you will leak C++ memory.
func LoadNN(baseFilename string) (*Orient, error) {
	// Convert Go string to C string
	cBaseFilename := C.CString(baseFilename)
	defer C.free(unsafe.Pointer(cBaseFilename))
	cOrient := C.LoadOrientationNN(cBaseFilename)
	if cOrient == nil {
		return nil, fmt.Errorf("failed to load neural network")
	}
	// Create a new Orient struct
	orient := &Orient{
		nn: cOrient,
	}
	return orient, nil
}

// Run the orientation network on a single 32x32 grayscale (8-bit) image.
// orientation returned is one of 0,1,2,3 for 0,90,180,270 degrees.
// confidence is a value between 0 and 1 indicating the confidence of the prediction.
func (o *Orient) Run(image []byte, width, height int) (orientation int, confidence float32, err error) {
	if width != 32 || height != 32 {
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
	if o.nn != nil {
		C.FreeOrientationNN(o.nn)
		o.nn = nil
	}
}
