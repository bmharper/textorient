package main

import (
	"fmt"
	"os"

	"github.com/bmharper/cimg/v2"
	"github.com/bmharper/textorient"
)

// Note that this does not straighten the image before determining orientation

func check(err error) {
	if err != nil {
		panic(err)
	}
}

func main() {
	inputFilename := os.Args[1]
	raw, err := cimg.ReadFile(inputFilename)
	check(err)
	orient, err := textorient.NewOrient()
	check(err)
	defer orient.Close()

	angle, err := orient.GetImageOrientation(raw)
	check(err)
	fmt.Printf("Orientation: %d\n", angle*90)
}
