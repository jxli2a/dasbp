## To install the processing code run:
```
git submodule update --init --recursive util/DAS-utilities/external/pybind11
cd util/DAS-utilities
mkdir build
cd build
cmake  -DCMAKE_INSTALL_PREFIX=../local ..
make install
```

## If the default cmake does not work, specify python environment manually
```
cmake -DPYTHON_EXECUTABLE={} -DPYTHON_INCLUDE_DIR={} -DPYTHON_LIBRARY={} -DCMAKE_INSTALL_PREFIX=../local ..
```
## Back-projection requires pycuda
## Interactive GUI requires ipywidgets
