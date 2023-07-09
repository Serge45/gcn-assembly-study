#!/bin/bash
objs=()
arch=$1

for i in "16 16" "8 32" "4 64" "2 128" "1 256" "2 256" "4 256" "8 256"; do
    set -- $i
    s=S_$1_$2.s
    o=S_$1_$2.o
    python3 ./codegen.py -o $s -m $1 -n $2 --arch $arch
    objs+=($o)
done

/opt/rocm/llvm/bin/clang++ -target amdgcn-amdhsa -o softmax_$arch.co ${objs[@]}