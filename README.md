## Brief ##
This repository collects some projects for my GCN assembly learning roads.

## How to Build
Clone this repository, `cd` to it, and then

```bash
mkdir build
cmake .. -DCMAKE_CXX_COMPILER=hipcc -DCMAKE_C_COMPILER=hipcc -DCMAKE_PREFIX_PATH=/opt/rocm/lib/cmake
make -j
```