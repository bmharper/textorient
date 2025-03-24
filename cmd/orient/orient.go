package main

import (
	"fmt"
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
	raw, err := cimg.ReadFile(inputFilename)
	check(err)
	orient, err := textorient.LoadNN("text_angle_classifier")
	check(err)
	defer orient.Close()

	angle, err := orient.RunImage(raw)
	check(err)
	fmt.Printf("Orientation: %d\n", angle*90)
}
