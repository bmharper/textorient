package textorient

import (
	"math"
	"testing"

	"github.com/bmharper/cimg/v2"
	"github.com/stretchr/testify/require"
)

func TestOrient(t *testing.T) {
	orient, err := NewOrient()
	if err != nil {
		t.Fatalf("NewOrient failed: %v", err)
	}
	defer orient.Close()

	// The raw image is upright.
	// We rotate it by 0,90,180,270 degrees and verify that at each rotation, we get the expected value out of the NN
	image, err := cimg.ReadFile("testdata/00001.jpg")
	require.NoError(t, err)
	image = image.ToGray()
	for iangle := 0; iangle <= 3; iangle++ {
		angle := float64(iangle) * 90
		copy := cimg.NewImage(image.Width, image.Height, image.Format)
		cimg.Rotate(image, copy, angle*math.Pi/180, nil)
		orientation, confidence, err := orient.getTileOrientation(copy.Pixels, copy.Width, copy.Height)
		if err != nil {
			t.Fatalf("Run failed: %v", err)
		}
		t.Logf("Orientation: %d, Confidence: %f", orientation, confidence)
		require.Equal(t, iangle, orientation)
		require.GreaterOrEqual(t, confidence, float32(0.5))
	}
}
