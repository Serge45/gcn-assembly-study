from typing import Optional, List, Tuple
from contextlib import contextmanager
from collections import OrderedDict
import yaml
from io import StringIO
from enum import Enum
import subprocess
import tempfile
from dataclasses import dataclass
import math

DEFAULT_CLANG_PATH = "/opt/rocm/llvm/bin/clang++"


class Gpr:
    gpr_type: str = None

    def __init__(self, idx: Optional[int]):
        self.index = idx

    def __str__(self):
        return f"{self.gpr_type}[{self.index}]"


class GprRange:
    gpr_type: str = None

    def __init__(self, index: int, size: int):
        self.index = index
        self.size = size

    def __str__(self):
        return f"{self.gpr_type}[{self.index}:{self.index+self.size-1}]"


class VgprRange(GprRange):
    gpr_type: str = "v"


class SgprRange(GprRange):
    gpr_type: str = "s"


class AccVgprRange(GprRange):
    gpr_type: str = "s"


class Vgpr(Gpr):
    gpr_type: str = "v"


class Sgpr(Gpr):
    gpr_type: str = "s"


class AccVgpr(Gpr):
    gpr_type: str = "acc"


class GprPool:
    gpr_type: str = None

    def __init__(self, size: int):
        self.size = size


class VgprPool(GprPool):
    gpr_type: str = "v"

    def __init__(self, size: int):
        super().__init__(size)
        self.pool = [i for i in range(size)]


class FunctionArgument:
    def __init__(self, typename: str, name: str, offset: Optional[int], num_bytes: int):
        self.typename = typename
        self.name = name
        self.offset = offset
        self.num_bytes = num_bytes
        self.address_space = "global" if typename == "global_buffer" else None


FunctionArgumentList = List[FunctionArgument]


def iter_kern_args(args: FunctionArgumentList):
    offset = 0
    for arg in args:
        yield (arg, arg.num_bytes, offset)
        offset += arg.offset


class FunctionMeta:
    def __init__(self, name: str, args: FunctionArgumentList):
        self.name = name
        self.kernarg_segment_size = 0
        self.group_segment_fixed_size = 0
        self.private_segment_fixed_size = 0
        self.kernarg_segment_align = 8
        self.wavefront_size = 64
        self.workgroup_size = 256
        self.sgpr_count = 0
        self.vgpr_count = 0
        self.agpr_count = 0
        self.args = args

    def normalized_args(self):
        offset = 0
        ret = []

        for arg in self.args:
            ret.append(
                {
                    ".size": arg.num_bytes,
                    ".offset": offset,
                    ".value_kind": arg.typename,
                    ".name": arg.name,
                }
            )

            if arg.address_space:
                ret[-1][".address_space"] = arg.address_space

            offset += arg.num_bytes

        self.kernarg_segment_size = offset
        return ret

    @property
    def argument_num_bytes(self):
        return sum(arg.num_bytes for arg in self.args)

    @property
    def argument_num_sgpr(self):
        return self.argument_num_bytes // 4

    def ro_data(self):
        return f"""
.rodata
.p2align 6
.amdhsa_kernel {self.name}
  .amdhsa_user_sgpr_kernarg_segment_ptr 1
  .amdhsa_system_sgpr_workgroup_id_x 1
  .amdhsa_system_sgpr_workgroup_id_y 1
  .amdhsa_accum_offset {max(self.vgpr_count//4*4, 4)}
  .amdhsa_group_segment_fixed_size {self.group_segment_fixed_size}
  .amdhsa_next_free_vgpr {self.vgpr_count}
  .amdhsa_next_free_sgpr {self.sgpr_count}
.end_amdhsa_kernel
"""

    def __str__(self):
        args = self.normalized_args()
        ret = {
            "amdhsa.version": [1, 1],
            "amdhsa.kernels": [
                {
                    ".name": self.name,
                    ".symbol": f"{self.name}.kd",
                    ".kernarg_segment_size": self.kernarg_segment_size,
                    ".group_segment_fixed_size": self.group_segment_fixed_size,
                    ".private_segment_fixed_size": self.private_segment_fixed_size,
                    ".kernarg_segment_align": self.kernarg_segment_align,
                    ".wavefront_size": self.wavefront_size,
                    ".max_flat_workgroup_size": self.workgroup_size,
                    ".sgpr_count": self.sgpr_count,
                    ".vgpr_count": self.vgpr_count,
                    ".agpr_count": self.agpr_count,
                    ".args": args,
                }
            ],
        }

        return f".amdgpu_metadata\n---\n{yaml.dump(ret)}\n.end_amdgpu_metadata"


def count_gprs(f):
    def wrapper(self, *args, **kwargs):
        for arg in args:
            if isinstance(arg, Vgpr):
                self.vgpr_counter = max(self.vgpr_counter, arg.index + 1)
                self.max_vgpr = max(self.max_vgpr, self.vgpr_counter)
            elif isinstance(arg, VgprRange):
                self.vgpr_counter = max(self.vgpr_counter, arg.index + arg.size + 1)
                self.max_vgpr = max(self.max_vgpr, self.vgpr_counter)
            elif isinstance(arg, Sgpr):
                self.sgpr_counter = max(self.sgpr_counter, arg.index + 1)
                self.max_sgpr = max(self.max_sgpr, self.sgpr_counter)
            elif isinstance(arg, SgprRange):
                self.sgpr_counter = max(self.sgpr_counter, arg.index + arg.size + 1)
                self.max_sgpr = max(self.max_sgpr, self.sgpr_counter)
            elif isinstance(arg, AccVgpr):
                self.agpr_counter = max(self.agpr_counter, arg.index + 1)
                self.max_agpr = max(self.max_agpr, self.agpr_counter)
            elif isinstance(arg, AccVgprRange):
                self.agpr_counter = max(self.agpr_counter, arg.index + arg.size + 1)
                self.max_agpr = max(self.max_agpr, self.agpr_counter)

        return f(self, *args, **kwargs)

    return wrapper


class GpuContext:
    def __init__(self):
        self.content = StringIO()
        self.instructions = []
        self.sgpr_counter = 0
        self.vgpr_counter = 0
        self.agpr_counter = 0
        self.max_sgpr = 0
        self.max_vgpr = 0
        self.max_agpr = 0

    def label(self, name: str):
        self.instructions.append([lambda: f"label_{name}:"])

    def comment(self, comment: str):
        self.instructions.append([lambda: f"//{comment}"])

    @count_gprs
    def buffer_load_dword(
        self,
        dst: Vgpr,
        voffset: Vgpr,
        srd: SgprRange,
        soffset: Sgpr | int,
        const_offset: int,
    ):
        self.instructions.append(
            [
                lambda: f"buffer_load_dword {str(dst)}, {str(voffset)}, {str(srd)}, {str(soffset)} offen offset:{const_offset}",
                dst,
                voffset,
                srd,
                const_offset,
            ]
        )

    @count_gprs
    def buffer_load_dwordx2(
        self,
        dst: VgprRange,
        voffset: Vgpr,
        srd: SgprRange,
        soffset: Sgpr | int,
        const_offset: int,
    ):
        self.instructions.append(
            [
                lambda: f"buffer_load_dwordx2 {str(dst)}, {str(voffset)}, {str(srd)}, {str(soffset)} offen offset:{const_offset}",
                dst,
                voffset,
                srd,
                const_offset,
            ]
        )

    @count_gprs
    def buffer_load_dwordx4(
        self,
        dst: VgprRange,
        voffset: Vgpr,
        srd: SgprRange,
        soffset: Sgpr | int,
        const_offset: int,
    ):
        self.instructions.append(
            [
                lambda: f"buffer_load_dwordx4 {str(dst)}, {str(voffset)}, {str(srd)}, {str(soffset)} offen offset:{const_offset}",
                dst,
                voffset,
                srd,
                const_offset,
            ]
        )

    def buffer_load_inst(self, num_dwords: int):
        if num_dwords == 1:
            return self.buffer_load_dword
        elif num_dwords == 2:
            return self.buffer_load_dwordx2
        elif num_dwords == 4:
            return self.buffer_load_dwordx4

    @count_gprs
    def buffer_store_dword(
        self,
        data: Vgpr,
        voffset: Vgpr,
        srd: SgprRange,
        soffset: Sgpr | int,
        const_offset: int,
    ):
        self.instructions.append(
            [
                lambda: f"buffer_store_dword {str(data)}, {str(voffset)}, {str(srd)}, {str(soffset)} offen offset:{soffset}",
                data,
                voffset,
                srd,
                const_offset,
            ]
        )

    @count_gprs
    def ds_write_b32(self, dst: Vgpr, voffset: Vgpr, const_offset: int):
        self.instructions.append(
            [
                lambda: f"ds_write_b32 {str(dst)}, {str(voffset)}, offset:{const_offset}",
                dst,
                voffset,
                const_offset,
            ]
        )

    @count_gprs
    def ds_write_b64(self, dst: Vgpr, vdata: VgprRange, const_offset: int):
        self.instructions.append(
            [
                lambda: f"ds_write_b64 {str(dst)}, {str(vdata)}, offset:{const_offset}",
                dst,
                vdata,
                const_offset,
            ]
        )

    @count_gprs
    def ds_write_b128(self, dst: Vgpr, vdata: VgprRange, const_offset: int):
        self.instructions.append(
            [
                lambda: f"ds_write_b128 {str(dst)}, {str(vdata)}, offset:{const_offset}",
                dst,
                vdata,
                const_offset,
            ]
        )

    def ds_write_inst(self, num_bytes):
        if num_bytes == 4:
            return self.ds_write_b32
        elif num_bytes == 8:
            return self.ds_write_b64
        elif num_bytes == 16:
            return self.ds_write_b128

    @count_gprs
    def ds_read_b32(self, dst: Vgpr, voffset: Vgpr, const_offset: int):
        self.instructions.append(
            [
                lambda: f"ds_read_b32 {str(dst)}, {str(voffset)}, offset:{const_offset}",
                dst,
                voffset,
                const_offset,
            ]
        )

    @count_gprs
    def ds_read_b64(self, dst: VgprRange, voffset: Vgpr, const_offset: int):
        self.instructions.append(
            [
                lambda: f"ds_read_b64 {str(dst)}, {str(voffset)}, offset:{const_offset}",
                dst,
                voffset,
                const_offset,
            ]
        )

    @count_gprs
    def ds_read_b128(self, dst: VgprRange, voffset: Vgpr, const_offset: int):
        self.instructions.append(
            [
                lambda: f"ds_read_b128 {str(dst)}, {str(voffset)}, offset:{const_offset}",
                dst,
                voffset,
                const_offset,
            ]
        )

    @count_gprs
    def s_mov_b32(self, dst: Sgpr, src: Sgpr | int | float):
        self.instructions.append(
            [
                lambda: f"s_mov_b32 {str(dst)}, {str(src)}",
                dst,
                src,
            ]
        )

    @count_gprs
    def s_mov_b64(self, dst: SgprRange, src: SgprRange):
        self.instructions.append(
            [
                lambda: f"s_mov_b64 {str(dst)}, {str(src)}",
                dst,
                src,
            ]
        )

    @count_gprs
    def s_lshl_b32(self, dst: Sgpr, src: Sgpr, shift: int):
        self.instructions.append(
            [lambda: f"s_lshl_b32 {str(dst)}, {str(src)} {shift}", dst, src, shift]
        )

    @count_gprs
    def s_mul_i32(self, dst: Sgpr, src0: Sgpr, src1: int | Sgpr):
        self.instructions.append(
            [lambda: f"s_mul_i32 {str(dst)}, {str(src0)} {str(src1)}", dst, src0, src1]
        )

    @count_gprs
    def s_add_i32(self, dst: Sgpr, src0: Sgpr, src1: int | Sgpr):
        self.instructions.append(
            [lambda: f"s_add_i32 {str(dst)}, {str(src0)} {str(src1)}", dst, src0, src1]
        )

    @count_gprs
    def s_load_dword(self, dst: Sgpr, src: SgprRange, offset: int):
        self.instructions.append(
            [lambda: f"s_load_dword {str(dst)}, {str(src)} {offset}", dst, src, offset]
        )

    @count_gprs
    def s_load_dwordx2(self, dst: SgprRange, src: SgprRange, offset: int):
        self.instructions.append(
            [
                lambda: f"s_load_dwordx2 {str(dst)}, {str(src)} {offset}",
                dst,
                src,
                offset,
            ]
        )

    @count_gprs
    def s_load_dwordx4(self, dst: SgprRange, src: SgprRange, offset: int):
        self.instructions.append(
            [
                lambda: f"s_load_dwordx4 {str(dst)}, {str(src)} {offset}",
                dst,
                src,
                offset,
            ]
        )

    def s_waitcnt(self, vmcnt: int = None, lgkmcnt: int = None):
        assert (vmcnt, lgkmcnt) != (None, None)

        def impl():
            args = []

            if vmcnt is not None:
                args.append(f"vmcnt({vmcnt})")

            if lgkmcnt is not None:
                args.append(f"lgkmcnt({lgkmcnt})")

            return " ".join(
                [
                    "s_waitcnt",
                ]
                + args
            )

        self.instructions.append([impl, vmcnt, lgkmcnt])

    def s_barrier(self):
        self.instructions.append([lambda: "s_barrier"])

    def s_endpgm(self):
        self.instructions.append([lambda: "s_endpgm"])

    @count_gprs
    def v_mov_b32(self, dst: Vgpr, src: Sgpr | Vgpr | int | float):
        self.instructions.append(
            [lambda: f"v_mov_b32 {str(dst)}, {str(src)}", dst, src]
        )

    @count_gprs
    def v_and_b32(
        self, dst: Vgpr, src0: Vgpr | int | float, src1: Sgpr | Vgpr | int | float
    ):
        self.instructions.append(
            [lambda: f"v_and_b32 {str(dst)}, {str(src0)}, {str(src1)}", dst, src0, src1]
        )

    @count_gprs
    def v_add_u32(
        self, dst: Vgpr, src0: Vgpr | int | float, src1: Sgpr | Vgpr | int | float
    ):
        self.instructions.append(
            [lambda: f"v_add_u32 {str(dst)}, {str(src0)}, {str(src1)}", dst, src0, src1]
        )

    @count_gprs
    def v_lshlrev_b32(
        self, dst: Vgpr, shift: Vgpr | int | float, src: Sgpr | Vgpr | int | float
    ):
        self.instructions.append(
            [
                lambda: f"v_lshlrev_b32 {str(dst)}, {str(shift)}, {str(src)}",
                dst,
                shift,
                src,
            ]
        )

    @count_gprs
    def v_lshrrev_b32(
        self, dst: Vgpr, shift: Vgpr | int | float, src: Sgpr | Vgpr | int | float
    ):
        self.instructions.append(
            [
                lambda: f"v_lshrrev_b32 {str(dst)}, {str(shift)}, {str(src)}",
                dst,
                shift,
                src,
            ]
        )

    @count_gprs
    def v_mul_lo_u32(self, dst: Vgpr, src0: Vgpr | int, src1: Sgpr | Vgpr | int):
        self.instructions.append(
            [
                lambda: f"v_mul_lo_u32 {str(dst)}, {str(src0)}, {str(src1)}",
                dst,
                src0,
                src1,
            ]
        )

    @count_gprs
    def v_mov_b64(self, dst: VgprRange, src: VgprRange | int | float):
        self.instructions.append(
            [lambda: f"v_mov_b64 {str(dst)}, {str(src)}", dst, src]
        )

    @count_gprs
    def v_accvgpr_write_b32(self, dst: AccVgpr, src: int | Vgpr):
        self.instructions.append(
            [lambda: f"v_accvgpr_write_b32 {str(dst)}, {str(src)}", dst, src]
        )

    def materialize(self):
        return "\n".join([inst[0]() for inst in self.instructions])


def gpu_function(func):
    def wrapper(context, *args, **kwargs):
        if context is None:
            context = GpuContext()
        return func(context, *args, **kwargs)

    return wrapper


class DataType(Enum):
    FP32 = 0


def datatype_size(dtype: DataType):
    if dtype == DataType.FP32:
        return 4

    assert False, "unrecognized type"


class GemmSolutionConfig:
    def __init__(
        self,
        a_type: DataType,
        b_type: DataType,
        cd_type: DataType,
        scalar_type: DataType,
        mfma: Tuple[int, int, int, int],
        wave_group: Tuple[int, int],
        wave_tiling: Tuple[int, int],
        depth_k: int,
        trans_a: bool,
        trans_b: bool,
    ):
        self.a_type = a_type
        self.b_type = b_type
        self.cd_type = cd_type
        self.scalar_type = scalar_type
        self.wave_group = wave_group
        self.trans_a = trans_a
        self.trans_b = trans_b
        self.mfma = mfma
        self.wave_tiling = wave_tiling
        self.depth_k = depth_k
        self.wavefront_size = 64

    @property
    def tile_size(self) -> Tuple[int, int]:
        return (
            self.mfma[0] * self.wave_group[0] * self.wave_tiling[0],
            self.mfma[1] * self.wave_group[1] * self.wave_tiling[1],
        )

    @property
    def num_workitems(self):
        return self.wave_group[0] * self.wave_group[1] * self.wavefront_size

    @property
    def num_bytes_per_buffer_load(self) -> Tuple[int, int]:
        t0, t1 = self.tile_size

        def num_bytes_loads(t, k, dtype):
            num_bytes = t * k * datatype_size(dtype)
            assert num_bytes % self.num_workitems == 0
            return num_bytes // self.num_workitems

        num_bytes_load_a = num_bytes_loads(t0, self.depth_k, self.a_type)
        num_bytes_load_b = num_bytes_loads(t1, self.depth_k, self.b_type)
        return min(num_bytes_load_a, 16), min(num_bytes_load_b, 16)

    @property
    def num_dwords_per_buffer_load(self) -> Tuple[int, int]:
        NUM_BYTES_DWORD = 4
        b0, b1 = self.num_bytes_per_buffer_load()
        return b0 // NUM_BYTES_DWORD, b1 // NUM_BYTES_DWORD

    @property
    def num_elements_per_ds_read(self) -> Tuple[int, int]:
        # TODO: support other MFMA
        return 1, 1

    @property
    def num_bytes_per_ds_read(self) -> Tuple[int, int]:
        return (
            self.num_elements_per_ds_read[0] * datatype_size(self.a_type),
            self.num_elements_per_ds_read[1] * datatype_size(self.b_type),
        )

    @property
    def lds_offset_bytes(self) -> Tuple[int, int]:
        return 0, self.tile_size[0] * self.depth_k * datatype_size(self.a_type)

    @property
    def lds_swap_offset_bytes(self) -> int:
        return self.tile_size[0] * self.depth_k * datatype_size(
            self.a_type
        ) + self.tile_size[1] * self.depth_k * datatype_size(self.b_type)

    @property
    def lds_usage_bytes(self) -> int:
        return 2 * (
            self.tile_size[0] * self.depth_k * datatype_size(self.a_type)
            + self.tile_size[1] * self.depth_k * datatype_size(self.b_type)
        )


@gpu_function
def gemm(
    context: GpuContext,
    name: str,
    arch: str,
    config: GemmSolutionConfig,
    arguments: FunctionArgumentList,
) -> str:
    meta = FunctionMeta(name, arguments)
    meta.group_segment_fixed_size = config.lds_usage_bytes

    @dataclass
    class SgprAlloc:
        srd_a: int
        srd_b: int
        srd_c: int
        srd_d: int
        wg_id_x: int
        wg_id_y: int
        m: int
        n: int
        k: int
        k_idx: int
        row_idx: int
        col_idx: int
        gl_offset_a: int
        gl_offset_b: int
        stride_a_0: int
        stride_a_1: int
        stride_b_0: int
        stride_b_1: int
        stride_c_0: int
        stride_c_1: int
        stride_d_0: int
        stride_d_1: int
        alpha: int
        beta: int
        lds_start_addr: int
        kern_args: int
        end: int

    @dataclass
    class VgprAlloc:
        t_id: int
        gl_offset_a: List[List[int]]
        gl_offset_b: List[List[int]]
        gl_offset_c: List[List[int]]
        gl_offset_d: List[List[int]]
        t_row: int
        t_col: int
        gl_data_a: List[List[int]]
        gl_data_b: List[List[int]]
        lw_addr_a: List[List[int]]
        lw_addr_b: List[List[int]]
        lr_addr_a: List[List[int]]
        lr_addr_b: List[List[int]]
        valu_a: List[List[int]]
        valu_b: List[List[int]]
        w_id: int
        w_row: int
        w_col: int

    @dataclass
    class AgprAlloc:
        num_reg_per_thread: int
        arpgs: List[List[List[int]]]

    def sgpr_alloc():
        # TODO: need clearer way to avoid overlapping
        return SgprAlloc(
            srd_a=4,
            srd_b=8,
            srd_c=4,
            srd_d=8,
            wg_id_x=2,
            wg_id_y=3,
            m=12,
            n=13,
            k=14,
            k_idx=15,
            row_idx=16,
            col_idx=17,
            gl_offset_a=18,
            gl_offset_b=19,
            stride_a_0=20,
            stride_a_1=21,
            stride_b_0=22,
            stride_b_1=23,
            stride_c_0=24,
            stride_c_1=25,
            stride_d_0=26,
            stride_d_1=27,
            alpha=28,
            beta=29,
            lds_start_addr=30,
            kern_args=32,
            end=32 + meta.argument_num_sgpr,
        )

    def vgpr_alloc():
        # TODO: for non-NN transposes, swap mt0, mt1 if required
        mt0, mt1 = config.tile_size
        depth_k = config.depth_k
        num_workitems = config.num_workitems

        vgpr_counter = 0
        t_id = 0
        vgpr_counter += 1

        glvw_bytes_a, glvw_bytes_b = config.num_bytes_per_buffer_load
        num_loads_a_0 = max(
            (mt0 * datatype_size(config.a_type)) // (glvw_bytes_a * num_workitems), 1
        )
        num_loads_a_1 = depth_k // (
            num_workitems // ((mt0 * datatype_size(config.a_type)) // glvw_bytes_a)
        )
        num_loads_b_0 = max(
            (depth_k * datatype_size(config.b_type)) // (glvw_bytes_b * num_workitems),
            1,
        )
        num_loads_b_1 = mt1 // (
            num_workitems // ((depth_k * datatype_size(config.b_type)) // glvw_bytes_b)
        )
        assert (num_loads_a_0, num_loads_a_1) != (0, 0)
        assert (num_loads_b_0, num_loads_b_1) != (0, 0)
        print(f"num_lods_a: {(num_loads_a_0, num_loads_a_1)}")
        print(f"num_lods_b: {(num_loads_b_0, num_loads_b_1)}")

        def gl_read_data(num_loads_0, num_loads_1, vw_num_vgpr):
            nonlocal vgpr_counter
            gl_datas = []

            for j in range(num_loads_1):
                indices = []
                for i in range(num_loads_0):
                    indices.append(vgpr_counter + vw_num_vgpr * (i + j * num_loads_0))
                gl_datas.append(indices)
            vgpr_counter += vw_num_vgpr * num_loads_0 * num_loads_1
            return gl_datas

        if (glvw_num_vgpr_a := (glvw_bytes_a // 4)) > 1:
            vgpr_counter = (vgpr_counter + 1) // 2 * 2

        gl_data_a = gl_read_data(num_loads_a_0, num_loads_a_1, glvw_num_vgpr_a)

        if (glvw_num_vgpr_b := (glvw_bytes_b // 4)) > 1:
            vgpr_counter = (vgpr_counter + 1) // 2 * 2

        gl_data_b = gl_read_data(num_loads_b_0, num_loads_b_1, glvw_num_vgpr_b)

        print("gl data vgpr:")
        print(gl_data_a)
        print(gl_data_b)

        # vw == 1 since we only need 1 VGPR to store offset for each thread
        gl_voffset_a = gl_read_data(num_loads_a_0, num_loads_a_1, 1)
        gl_voffset_b = gl_read_data(num_loads_b_0, num_loads_b_1, 1)
        print("gl offset vgpr:")
        print(gl_voffset_a)
        print(gl_voffset_b)

        lw_voffset_a = gl_read_data(num_loads_a_0, num_loads_a_1, 1)
        lw_voffset_b = gl_read_data(num_loads_b_0, num_loads_b_1, 1)
        print("lw offset vgpr:")
        print(lw_voffset_a)
        print(lw_voffset_b)

        lr_addr_a = gl_read_data(config.wave_tiling[0], 1, 1)
        lr_addr_b = gl_read_data(1, config.wave_tiling[1], 1)

        print("ds read addr")
        print(lr_addr_a)
        print(lr_addr_b)
        valu_num_vgpr_a = config.num_bytes_per_ds_read[0] // 4
        valu_num_vgpr_b = config.num_bytes_per_ds_read[1] // 4

        valu_a = gl_read_data(config.wave_tiling[0], 1, valu_num_vgpr_a)
        valu_b = gl_read_data(1, config.wave_tiling[1], valu_num_vgpr_b)

        print("valu")
        print(valu_a)
        print(valu_b)

        w_id = vgpr_counter
        vgpr_counter += 1
        w_row = vgpr_counter
        vgpr_counter += 1
        w_col = vgpr_counter
        vgpr_counter += 1
        t_row = vgpr_counter
        vgpr_counter += 1
        t_col = vgpr_counter
        vgpr_counter += 1

        return VgprAlloc(
            t_id=t_id,
            gl_offset_a=gl_voffset_a,
            gl_offset_b=gl_voffset_b,
            gl_offset_c=None,  # TODO: add this
            gl_offset_d=None,  # TODO: add this
            t_row=t_row,
            t_col=t_col,
            gl_data_a=gl_data_a,
            gl_data_b=gl_data_b,
            lw_addr_a=lw_voffset_a,
            lw_addr_b=lw_voffset_b,
            lr_addr_a=lr_addr_a,
            lr_addr_b=lr_addr_b,
            valu_a=valu_a,
            valu_b=valu_b,
            w_id=w_id,
            w_row=w_row,
            w_col=w_col,
        )

    def agpr_alloc():
        agprs = AgprAlloc(4, [])

        for j in range(config.wave_tiling[1]):
            val = []
            for i in range(config.wave_tiling[0]):
                val.append(4 * (i + j * config.wave_tiling[0]))
            agprs.arpgs.append(val)

        return agprs

    @contextmanager
    def alloc_tmp_sgpr(num_regs: int):
        sgpr = (
            SgprRange(context.sgpr_counter, num_regs)
            if num_regs > 1
            else Sgpr(context.sgpr_counter)
        )
        context.sgpr_counter += num_regs
        context.max_sgpr = max(context.max_sgpr, context.sgpr_counter)
        try:
            yield sgpr
        finally:
            context.sgpr_counter -= num_regs

    @contextmanager
    def alloc_tmp_vgpr(num_regs: int):
        vgpr = (
            VgprRange(context.sgpr_counter, num_regs)
            if num_regs > 1
            else Vgpr(context.sgpr_counter)
        )
        context.vgpr_counter += num_regs
        context.max_vgpr = max(context.max_vgpr, context.vgpr_counter)
        try:
            yield vgpr
        finally:
            context.vgpr_counter -= num_regs

    def header():
        return f"""
.amdgcn_target "amdgcn-amd-amdhsa--{arch}"
.text
.globl {name}
.p2align 8
.type {name},@function
"""

    def implementation():
        sgprs = sgpr_alloc()
        num_sgpr_kernarg = meta.argument_num_sgpr
        kern_arg_sgpr_offset = 0
        context.label("load_args")
        context.comment("Load all arguments")
        while num_sgpr_kernarg:
            if num_sgpr_kernarg >= 4:
                context.s_load_dwordx4(
                    SgprRange(sgprs.kern_args + kern_arg_sgpr_offset, 4),
                    SgprRange(0, 2),
                    kern_arg_sgpr_offset * 4,
                )
                kern_arg_sgpr_offset += 4
                num_sgpr_kernarg -= 4
            elif num_sgpr_kernarg >= 2:
                context.s_load_dwordx2(
                    SgprRange(sgprs.kern_args + kern_arg_sgpr_offset, 2),
                    SgprRange(0, 2),
                    kern_arg_sgpr_offset * 4,
                )
                kern_arg_sgpr_offset += 2
                num_sgpr_kernarg -= 2
            else:
                context.s_load_dword(
                    Sgpr(sgprs.kern_args + kern_arg_sgpr_offset),
                    SgprRange(0, 2),
                    kern_arg_sgpr_offset * 4,
                )
                kern_arg_sgpr_offset += 1
                num_sgpr_kernarg -= 1

        context.comment("Setup Srd{A, B}")
        context.s_mov_b32(Sgpr(sgprs.srd_a + 3), 0x20000)
        context.s_mov_b32(Sgpr(sgprs.srd_b + 3), 0x20000)
        context.s_waitcnt(lgkmcnt=0)
        context.s_mov_b64(SgprRange(sgprs.srd_a, 2), SgprRange(sgprs.kern_args, 2))
        context.s_mov_b64(SgprRange(sgprs.srd_b, 2), SgprRange(sgprs.kern_args + 2, 2))
        context.comment("Setup sizes, m, n and k")
        context.s_mov_b32(Sgpr(sgprs.m), Sgpr(sgprs.kern_args + 8))
        context.s_mov_b32(Sgpr(sgprs.n), Sgpr(sgprs.kern_args + 9))
        context.s_mov_b32(Sgpr(sgprs.k), Sgpr(sgprs.kern_args + 10))

        context.comment("Setup global read offsets")
        context.s_lshl_b32(Sgpr(sgprs.row_idx), Sgpr(sgprs.wg_id_x), 5)

        bpe_log_a = int(math.log2(datatype_size(gemm_config.a_type)))
        bpe_log_b = int(math.log2(datatype_size(gemm_config.b_type)))

        with alloc_tmp_sgpr(1) as tmp:
            context.s_mul_i32(tmp, Sgpr(sgprs.m), Sgpr(sgprs.k))
            context.s_lshl_b32(Sgpr(sgprs.srd_a + 2), tmp, bpe_log_a)
            context.s_mul_i32(tmp, Sgpr(sgprs.n), Sgpr(sgprs.k))
            context.s_lshl_b32(Sgpr(sgprs.srd_b + 2), tmp, bpe_log_b)

        context.s_mov_b32(Sgpr(sgprs.stride_a_0), 1)
        context.s_mov_b32(Sgpr(sgprs.stride_a_1), Sgpr(sgprs.kern_args + 11))
        context.s_mov_b32(Sgpr(sgprs.stride_b_0), 1)
        context.s_mov_b32(Sgpr(sgprs.stride_b_1), Sgpr(sgprs.kern_args + 12))
        context.s_mov_b32(Sgpr(sgprs.stride_c_0), 1)
        context.s_mov_b32(Sgpr(sgprs.stride_c_1), Sgpr(sgprs.kern_args + 13))
        context.s_mov_b32(Sgpr(sgprs.stride_d_0), 1)
        context.s_mov_b32(Sgpr(sgprs.stride_d_1), Sgpr(sgprs.kern_args + 14))
        context.s_mul_i32(
            Sgpr(sgprs.gl_offset_a), Sgpr(sgprs.row_idx), Sgpr(sgprs.stride_a_0)
        )
        context.s_lshl_b32(Sgpr(sgprs.gl_offset_a), Sgpr(sgprs.gl_offset_a), bpe_log_a)
        context.s_mul_i32(
            Sgpr(sgprs.gl_offset_b), Sgpr(sgprs.col_idx), Sgpr(sgprs.stride_b_1)
        )
        context.s_lshl_b32(Sgpr(sgprs.gl_offset_b), Sgpr(sgprs.gl_offset_b), bpe_log_b)
        context.s_mov_b32(Sgpr(sgprs.k_idx), 0)
        agprs = agpr_alloc()

        for col in agprs.arpgs:
            for row in col:
                for i in range(row, row + agprs.num_reg_per_thread):
                    context.v_accvgpr_write_b32(AccVgpr(i), 0)

        vgprs = vgpr_alloc()
        context.label("addr_calculations")
        gl_num_elements_a = config.num_bytes_per_buffer_load[0] // datatype_size(
            config.a_type
        )
        num_load_threads0_a = config.tile_size[0] // gl_num_elements_a
        num_load_threads1_a = config.num_workitems // num_load_threads0_a
        context.v_and_b32(
            Vgpr(vgprs.t_row),
            Vgpr(vgprs.t_id),
            config.tile_size[0] // gl_num_elements_a - 1,
        )
        context.v_mul_lo_u32(Vgpr(vgprs.t_row), Vgpr(vgprs.t_row), gl_num_elements_a)
        context.v_lshlrev_b32(
            Vgpr(vgprs.t_col), int(math.log2(num_load_threads0_a)), Vgpr(vgprs.t_col)
        )

        for j, col in enumerate(vgprs.gl_offset_a):
            for i, row in enumerate(col):
                context.comment(f"gl_addr_a_{j}_{i}")
                context.v_mul_lo_u32(
                    Vgpr(vgprs.gl_offset_a[j][i]),
                    Vgpr(vgprs.t_col),
                    Sgpr(sgprs.stride_a_1),
                )
                context.v_add_u32(
                    Vgpr(vgprs.gl_offset_a[j][i]),
                    Vgpr(vgprs.gl_offset_a[j][i]),
                    Vgpr(vgprs.t_row),
                )
                context.v_mul_lo_u32(
                    Vgpr(vgprs.gl_offset_a[j][i]),
                    datatype_size(config.a_type),
                    Vgpr(vgprs.gl_offset_a[j][i]),
                )
                context.v_add_u32(
                    Vgpr(vgprs.t_row),
                    num_load_threads0_a * gl_num_elements_a,
                    Vgpr(vgprs.t_row),
                )
                context.v_add_u32(
                    Vgpr(vgprs.t_col), num_load_threads1_a, Vgpr(vgprs.t_col)
                )

        gl_num_elements_b = config.num_bytes_per_buffer_load[1] // datatype_size(
            config.b_type
        )
        num_load_threads0_b = config.depth_k // gl_num_elements_b
        num_load_threads1_b = config.num_workitems // num_load_threads0_b
        context.v_and_b32(
            Vgpr(vgprs.t_row), Vgpr(vgprs.t_id), config.depth_k // gl_num_elements_b - 1
        )
        context.v_mul_lo_u32(Vgpr(vgprs.t_row), Vgpr(vgprs.t_row), gl_num_elements_b)
        context.v_lshlrev_b32(
            Vgpr(vgprs.t_col), int(math.log2(num_load_threads0_b)), Vgpr(vgprs.t_col)
        )

        for j, col in enumerate(vgprs.gl_offset_b):
            for i, row in enumerate(col):
                context.comment(f"gl_addr_b_{j}_{i}")
                context.v_mul_lo_u32(
                    Vgpr(vgprs.gl_offset_b[j][i]),
                    Vgpr(vgprs.t_col),
                    Sgpr(sgprs.stride_b_1),
                )
                context.v_add_u32(
                    Vgpr(vgprs.gl_offset_b[j][i]),
                    Vgpr(vgprs.gl_offset_b[j][i]),
                    Vgpr(vgprs.t_row),
                )
                context.v_mul_lo_u32(
                    Vgpr(vgprs.gl_offset_b[j][i]),
                    Vgpr(vgprs.gl_offset_b[j][i]),
                    datatype_size(config.b_type),
                )
                context.v_add_u32(
                    Vgpr(vgprs.t_row),
                    num_load_threads0_b * gl_num_elements_b,
                    Vgpr(vgprs.t_row),
                )
                context.v_add_u32(
                    Vgpr(vgprs.t_col), num_load_threads1_b, Vgpr(vgprs.t_col)
                )

        context.comment("gl_a")
        for j, col in enumerate(vgprs.gl_data_a):
            for i, row in enumerate(col):
                num_dwords_per_load = (
                    gl_num_elements_a * datatype_size(config.a_type) // 4
                )
                dst = (
                    VgprRange(row, num_dwords_per_load)
                    if num_dwords_per_load > 1
                    else Vgpr(row)
                )
                context.buffer_load_inst(num_dwords_per_load)(
                    dst,
                    Vgpr(vgprs.gl_offset_a[j][i]),
                    SgprRange(sgprs.srd_a, 4),
                    Sgpr(sgprs.gl_offset_a),
                    0,
                )

        context.comment("gl_b")
        for j, col in enumerate(vgprs.gl_data_b):
            for i, row in enumerate(col):
                num_dwords_per_load = (
                    gl_num_elements_b * datatype_size(config.b_type) // 4
                )
                dst = (
                    VgprRange(row, num_dwords_per_load)
                    if num_dwords_per_load > 1
                    else Vgpr(row)
                )
                context.buffer_load_inst(num_dwords_per_load)(
                    dst,
                    Vgpr(vgprs.gl_offset_b[j][i]),
                    SgprRange(sgprs.srd_b, 4),
                    Sgpr(sgprs.gl_offset_b),
                    0,
                )

        context.comment("lw_a")
        context.v_and_b32(
            Vgpr(vgprs.t_row),
            Vgpr(vgprs.t_id),
            config.tile_size[0] // gl_num_elements_a - 1,
        )

        context.v_mul_lo_u32(Vgpr(vgprs.t_row), Vgpr(vgprs.t_row), gl_num_elements_a)
        context.v_lshlrev_b32(
            Vgpr(vgprs.t_col), int(math.log2(num_load_threads0_a)), Vgpr(vgprs.t_col)
        )

        for j, col in enumerate(vgprs.lw_addr_a):
            for i, row in enumerate(col):
                with alloc_tmp_sgpr(1) as stmp:
                    context.s_mov_b32(stmp, config.tile_size[0])
                    context.v_mul_lo_u32(Vgpr(row), Vgpr(vgprs.t_col), stmp)
                context.v_add_u32(Vgpr(row), Vgpr(row), Vgpr(vgprs.t_row))
                context.v_mul_lo_u32(Vgpr(row), Vgpr(row), datatype_size(config.a_type))

        context.comment("lw_b")
        context.v_and_b32(
            Vgpr(vgprs.t_row), Vgpr(vgprs.t_id), config.depth_k // gl_num_elements_b - 1
        )
        context.v_mul_lo_u32(Vgpr(vgprs.t_row), Vgpr(vgprs.t_row), gl_num_elements_b)
        context.v_lshlrev_b32(
            Vgpr(vgprs.t_col), int(math.log2(num_load_threads0_b)), Vgpr(vgprs.t_col)
        )

        for j, col in enumerate(vgprs.lw_addr_b):
            for i, row in enumerate(col):
                if config.depth_k > 127:
                    with alloc_tmp_sgpr(1) as stmp:
                        context.s_mov_b32(stmp, config.depth_k)
                        context.v_mul_lo_u32(Vgpr(row), Vgpr(vgprs.t_col), stmp)
                else:
                    context.v_mul_lo_u32(Vgpr(row), Vgpr(vgprs.t_col), config.depth_k)
                context.v_add_u32(Vgpr(row), Vgpr(row), Vgpr(vgprs.t_row))
                context.v_mul_lo_u32(Vgpr(row), Vgpr(row), datatype_size(config.b_type))

        context.s_waitcnt(vmcnt=0)

        for j, col in enumerate(vgprs.lw_addr_a):
            for i, row in enumerate(col):
                vdata = (
                    VgprRange(
                        vgprs.gl_data_a[j][i], config.num_bytes_per_buffer_load[0] // 4
                    )
                    if config.num_bytes_per_buffer_load[0] > 4
                    else VgprRange(vgprs.gl_data_a[j][i])
                )
                context.ds_write_inst(config.num_bytes_per_buffer_load[0])(
                    Vgpr(row), vdata, config.lds_offset_bytes[0]
                )

        for j, col in enumerate(vgprs.lw_addr_b):
            for i, row in enumerate(col):
                vdata = (
                    VgprRange(
                        vgprs.gl_data_b[j][i], config.num_bytes_per_buffer_load[1] // 4
                    )
                    if config.num_bytes_per_buffer_load[1] > 4
                    else VgprRange(vgprs.gl_data_b[j][i])
                )
                context.ds_write_inst(config.num_bytes_per_buffer_load[1])(
                    Vgpr(row), vdata, config.lds_offset_bytes[1]
                )

        with alloc_tmp_sgpr(1) as tmp_sgpr:
            context.comment("gl_increment_a")
            context.s_mul_i32(
                tmp_sgpr,
                Sgpr(sgprs.stride_a_1),
                config.depth_k * datatype_size(config.a_type),
            )
            context.s_add_i32(
                Sgpr(sgprs.gl_offset_a), Sgpr(sgprs.gl_offset_a), tmp_sgpr
            )
            context.comment("gl_increment_b")
            context.s_mul_i32(
                tmp_sgpr,
                Sgpr(sgprs.stride_b_1),
                config.depth_k * datatype_size(config.b_type),
            )
            context.s_add_i32(
                Sgpr(sgprs.gl_offset_b), Sgpr(sgprs.gl_offset_b), tmp_sgpr
            )

            context.comment("swap ds write address")
            context.s_mov_b32(tmp_sgpr, config.lds_swap_offset_bytes)

            for j, col in enumerate(vgprs.lw_addr_a):
                for i, row in enumerate(col):
                    context.v_add_u32(Vgpr(row), Vgpr(row), tmp_sgpr)

            for j, col in enumerate(vgprs.lw_addr_b):
                for i, row in enumerate(col):
                    context.v_add_u32(Vgpr(row), Vgpr(row), tmp_sgpr)

        # TODO: ds read addresses
        # TODO: sync ds write and barrier
        # TODO: unrolled loop
        # TODO: load c
        # TODO: compute d = a*b+c
        # TODO: store d

        context.s_endpgm()
        return context.materialize()

    def body():
        return f"""
{name}:
{implementation()}
.L{name}_end:
    .size {name}, .L{name}_end - {name}
"""

    context.content.write(header())
    context.content.write(body())
    meta.sgpr_count = context.sgpr_counter
    meta.vgpr_count = context.vgpr_counter
    meta.agpr_count = context.agpr_counter
    context.content.write(meta.ro_data())
    context.content.write(str(meta))
    return context.content.getvalue()


gemm_config = GemmSolutionConfig(
    DataType.FP32,
    DataType.FP32,
    DataType.FP32,
    DataType.FP32,
    (16, 16, 1, 4),
    (2, 2),
    (1, 1),
    16,
    False,
    False,
)
print(gemm_config.tile_size, gemm_config.num_workitems)
print(gemm_config.num_bytes_per_buffer_load)

arch = "gfx90a:xnack-"
asm_str = gemm(
    None,
    "gemm",
    arch,
    gemm_config,
    [
        FunctionArgument("global_buffer", "a", None, 8),
        FunctionArgument("global_buffer", "b", None, 8),
        FunctionArgument("global_buffer", "c", None, 8),
        FunctionArgument("global_buffer", "d", None, 8),
        FunctionArgument("by_value", "m", None, 4),
        FunctionArgument("by_value", "n", None, 4),
        FunctionArgument("by_value", "k", None, 4),
        FunctionArgument("by_value", "lda", None, 4),
        FunctionArgument("by_value", "ldb", None, 4),
        FunctionArgument("by_value", "ldc", None, 4),
        FunctionArgument("by_value", "ldd", None, 4),
        FunctionArgument("by_value", "alpha", None, 4),
        FunctionArgument("by_value", "beta", None, 4),
    ],
)

print(asm_str)

with tempfile.NamedTemporaryFile("w", encoding="utf-8") as f:
    f.write(asm_str)
    f.flush()
    ret = subprocess.run(
        [
            DEFAULT_CLANG_PATH,
            "-x",
            "assembler",
            "-target",
            "amdgcn-amd-amdhsa",
            "-mcode-object-version=4",
            f"-mcpu={arch}",
            "-mwavefrontsize64",
            "-c",
            "-g",
            f.name,
            "-o",
            "test.o",
        ]
    )
    ret = subprocess.run(
        [DEFAULT_CLANG_PATH, "-target", "amdgcn-amd-amdhsa", "test.o", "-o", "test.co"]
    )
