package main

import (
	"os"

	"github.com/bmharper/cimg/v2"
	"github.com/bmharper/textorient"
)

func check(err error) {
	if err != nil {
		panic(err)
	}
}

func main() {
	inputFilename := os.Args[1]
	outputFilename := os.Args[2]
	org, err := cimg.ReadFile(inputFilename)
	check(err)

	orient, err := textorient.NewOrient("text_angle_classifier")
	check(err)

	straight, err := orient.StraightenImage(org, nil)
	check(err)
	straight.WriteJPEG(outputFilename, cimg.MakeCompressParams(cimg.Sampling444, 95, 0), 0644)
}
