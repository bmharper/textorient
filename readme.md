# textorient

textorient is a Go package that runs a small embedded NN (using NCNN) to determine
the orientation of text in an image. Our NN is trained to produce 1 of 4 outputs:

- 0: 0 degrees
- 1: 90 degrees
- 2: 180 degrees
- 3: 270 degrees

## Build

```
sudo apt install protobuf-compiler
```

```
cd ncnn
mkdir build
cd build
cmake -DNCNN_BUILD_TOOLS=ON -DNCNN_BUILD_EXAMPLES=OFF -DNCNN_BUILD_BENCHMARK=OFF -DNCNN_BUILD_TESTS=OFF ..
```
