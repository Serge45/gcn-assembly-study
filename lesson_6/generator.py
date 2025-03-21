from typing import Optional, List, Tuple, Dict
from contextlib import contextmanager
import yaml
from io import StringIO
from enum import IntEnum
import subprocess
import tomli_w
from dataclasses import dataclass
import math
from itertools import cycle, islice

def roundrobin(*iterables):
    iterators = map(iter, iterables)
    for num_active in range(len(iterables), 0, -1):
        iterators = cycle(islice(iterators, num_active))
        yield from map(next, iterators)

DEFAULT_CLANG_PATH = "/opt/rocm/llvm/bin/clang++"
MAX_LDS_NUM_BYTES = 65536


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

    def split(self) -> List[Gpr]:
        if isinstance(self, VgprRange):
            return [Vgpr(self.index+i) for i in range(self.size)]
        elif isinstance(self, SgprRange):
            return [Sgpr(self.index+i) for i in range(self.size)]
        elif isinstance(self, AccVgprRange):
            return [AccVgpr(self.index+i) for i in range(self.size)]

class VgprRange(GprRange):
    gpr_type: str = "v"


class SgprRange(GprRange):
    gpr_type: str = "s"


class AccVgprRange(GprRange):
    gpr_type: str = "acc"


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
  .amdhsa_accum_offset {max((self.vgpr_count+3)//4*4, 4)}
  .amdhsa_group_segment_fixed_size {self.group_segment_fixed_size}
  .amdhsa_next_free_vgpr {self.vgpr_count+self.agpr_count}
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

def count_calls(f):
    def wrapper(*args, **kwargs):
        wrapper._num_calls += 1
        return f(*args, **kwargs)
    wrapper._num_calls = 0
    return wrapper

def count_gprs(f):
    def wrapper(self, *args, **kwargs):
        for arg in args:
            if isinstance(arg, Vgpr):
                self.vgpr_counter = max(self.vgpr_counter, arg.index + 1)
                self.max_vgpr = max(self.max_vgpr, self.vgpr_counter)
            elif isinstance(arg, VgprRange):
                self.vgpr_counter = max(self.vgpr_counter, arg.index + arg.size)
                self.max_vgpr = max(self.max_vgpr, self.vgpr_counter)
            elif isinstance(arg, Sgpr):
                self.sgpr_counter = max(self.sgpr_counter, arg.index + 1)
                self.max_sgpr = max(self.max_sgpr, self.sgpr_counter)
            elif isinstance(arg, SgprRange):
                self.sgpr_counter = max(self.sgpr_counter, arg.index + arg.size)
                self.max_sgpr = max(self.max_sgpr, self.sgpr_counter)
            elif isinstance(arg, AccVgpr):
                self.agpr_counter = max(self.agpr_counter, arg.index + 1)
                self.max_agpr = max(self.max_agpr, self.agpr_counter)
            elif isinstance(arg, AccVgprRange):
                self.agpr_counter = max(self.agpr_counter, arg.index + arg.size)
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

    def label_name(self, name: str):
        return f"label_{name}"

    def label(self, name: str):
        self.instructions.append([lambda: f"{self.label_name(name)}:"])

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
                lambda: f"buffer_store_dword {str(data)}, {str(voffset)}, {str(srd)}, {str(soffset)} offen offset:{const_offset}",
                data,
                voffset,
                srd,
                const_offset,
            ]
        )

    @count_gprs
    def buffer_store_dwordx2(
        self,
        data: VgprRange,
        voffset: Vgpr,
        srd: SgprRange,
        soffset: Sgpr | int,
        const_offset: int,
    ):
        self.instructions.append(
            [
                lambda: f"buffer_store_dwordx2 {str(data)}, {str(voffset)}, {str(srd)}, {str(soffset)} offen offset:{const_offset}",
                data,
                voffset,
                srd,
                const_offset,
            ]
        )

    @count_gprs
    def buffer_store_dwordx4(
        self,
        data: VgprRange,
        voffset: Vgpr,
        srd: SgprRange,
        soffset: Sgpr | int,
        const_offset: int,
    ):
        self.instructions.append(
            [
                lambda: f"buffer_store_dwordx4 {str(data)}, {str(voffset)}, {str(srd)}, {str(soffset)} offen offset:{const_offset}",
                data,
                voffset,
                srd,
                const_offset,
            ]
        )

    def buffer_store_inst(self, num_dwords: int):
        if num_dwords == 1:
            return self.buffer_store_dword
        elif num_dwords == 2:
            return self.buffer_store_dwordx2
        elif num_dwords == 4:
            return self.buffer_store_dwordx4

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

    def ds_read_inst(self, num_bytes: int):
        if num_bytes == 4:
            return self.ds_read_b32
        elif num_bytes == 8:
            return self.ds_read_b64
        elif num_bytes == 16:
            return self.ds_read_b128

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
            [lambda: f"s_lshl_b32 {str(dst)}, {str(src)}, {shift}", dst, src, shift]
        )

    @count_gprs
    def s_lshr_b32(self, dst: Sgpr, src: Sgpr, shift: int):
        self.instructions.append(
            [lambda: f"s_lshr_b32 {str(dst)}, {str(src)}, {shift}", dst, src, shift]
        )

    @count_gprs
    def s_mul_i32(self, dst: Sgpr, src0: Sgpr, src1: int | Sgpr):
        self.instructions.append(
            [lambda: f"s_mul_i32 {str(dst)}, {str(src0)}, {str(src1)}", dst, src0, src1]
        )

    @count_gprs
    def s_add_i32(self, dst: Sgpr, src0: Sgpr, src1: int | Sgpr):
        self.instructions.append(
            [lambda: f"s_add_i32 {str(dst)}, {str(src0)}, {str(src1)}", dst, src0, src1]
        )

    @count_gprs
    def s_sub_i32(self, dst: Sgpr, src0: Sgpr, src1: int | Sgpr):
        self.instructions.append(
            [lambda: f"s_sub_i32 {str(dst)}, {str(src0)}, {str(src1)}", dst, src0, src1]
        )

    @count_gprs
    def s_and_b32(self, dst: Sgpr, src0: Sgpr, src1: int | Sgpr):
        self.instructions.append(
            [lambda: f"s_and_b32 {str(dst)}, {str(src0)}, {str(src1)}", dst, src0, src1]
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

    @count_calls
    @count_gprs
    def s_div_u32(self, dst: Sgpr, remainder: Sgpr, dividend: Sgpr, divisor: Sgpr):
        end_label_name = f"s_division_end_{self.s_div_u32._num_calls}"
        self.s_mov_b32(remainder, 0)
        self.s_cmp_eq_u32(dividend, divisor)
        self.s_cselect_b32(dst, 1, 0)
        self.s_cbranch_scc1(end_label_name)
        self.s_cmp_lt_u32(dividend, divisor)
        self.s_cselect_b32(dst, 0, 1)
        self.s_mov_b32(remainder, divisor)
        self.s_cbranch_scc1(end_label_name)
        div_beg_label_name = f"s_division_shift_{self.s_div_u32._num_calls}"
        div_end_label_name = f"s_division_shift_end_{self.s_div_u32._num_calls}"
        self.s_mov_b32(remainder, divisor)

        self.label(div_beg_label_name)
        self.s_cmp_lt_u32(dividend, remainder)
        self.s_cbranch_scc1(div_end_label_name)
        self.s_lshl_b32(dst, dst, 1)
        self.s_lshl_b32(remainder, remainder, 1)
        self.s_branch(div_beg_label_name)
        self.label(div_end_label_name)

        div_beg_sub_label_name = f"s_division_sub_{self.s_div_u32._num_calls}"
        div_end_sub_label_name = f"s_division_sub_end_{self.s_div_u32._num_calls}"
        self.label(div_beg_sub_label_name)
        self.s_cmp_le_u32(remainder, dividend)
        self.s_cbranch_scc1(div_end_sub_label_name)
        self.s_sub_i32(remainder, remainder, divisor)
        self.s_sub_i32(dst, dst, 1)
        self.s_branch(div_beg_sub_label_name)
        self.label(div_end_sub_label_name)
        self.s_sub_i32(remainder, dividend, remainder)
        self.label(end_label_name)

    @count_gprs
    def s_cmp_lt_u32(self, lhs: Sgpr | int, rhs: Sgpr | int):
        self.instructions.append([lambda: f"s_cmp_lt_u32 {str(lhs)}, {str(rhs)}", lhs, rhs])

    @count_gprs
    def s_cmp_le_u32(self, lhs: Sgpr | int, rhs: Sgpr | int):
        self.instructions.append([lambda: f"s_cmp_le_u32 {str(lhs)}, {str(rhs)}", lhs, rhs])

    @count_gprs
    def s_cmp_eq_u32(self, lhs: Sgpr | int, rhs: Sgpr | int):
        self.instructions.append([lambda: f"s_cmp_eq_u32 {str(lhs)}, {str(rhs)}", lhs, rhs])

    @count_gprs
    def s_cselect_b32(self, dst: Sgpr, lhs: Sgpr | int, rhs: Sgpr | int):
        self.instructions.append([lambda: f"s_cselect_b32 {str(dst)}, {str(lhs)}, {str(rhs)}", dst, lhs, rhs])

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

    def s_cbranch_scc1(self, name: str):
        self.instructions.append([lambda: f"s_cbranch_scc1 {self.label_name(name)}"])

    def s_branch(self, name: str):
        self.instructions.append([lambda: f"s_branch {self.label_name(name)}"])


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
        self, dst: Vgpr, src0: Vgpr | int, src1: Sgpr | Vgpr | int
    ):
        self.instructions.append(
            [lambda: f"v_add_u32 {str(dst)}, {str(src0)}, {str(src1)}", dst, src0, src1]
        )

    @count_gprs
    def v_add_i32(
        self, dst: Vgpr, src0: Vgpr | int, src1: Sgpr | Vgpr | int
    ):
        self.instructions.append(
            [lambda: f"v_add_i32 {str(dst)}, {str(src0)}, {str(src1)}", dst, src0, src1]
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
    def v_mul_f32(self, dst: Vgpr, src0: Vgpr | float, src1: Sgpr | Vgpr | float):
        self.instructions.append(
            [
                lambda: f"v_mul_f32 {str(dst)}, {str(src0)}, {str(src1)}",
                dst,
                src0,
                src1,
            ]
        )

    @count_gprs
    def v_fma_f32(self, dst: Vgpr, src0: Vgpr | float, src1: Sgpr | Vgpr | float, src2: Sgpr | Vgpr | float):
        self.instructions.append(
            [
                lambda: f"v_fma_f32 {str(dst)}, {str(src0)}, {str(src1)}, {str(src2)}",
                dst,
                src0,
                src1,
                src2,
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

    @count_gprs
    def v_accvgpr_read_b32(self, dst: Vgpr, src: AccVgpr):
        self.instructions.append(
            [lambda: f"v_accvgpr_read_b32 {str(dst)}, {str(src)}", dst, src]
        )

    @count_gprs
    def v_mfma_f32_16x16x4f32(self, acc: AccVgprRange, a: Vgpr, b: Vgpr, c: AccVgprRange):
        self.instructions.append(
            [lambda: f"v_mfma_f32_16x16x4f32 {str(acc)}, {str(a)}, {str(b)}, {str(c)}", acc, a, b, c]
        )

    def mfma_inst(self, mfma: Tuple[int, int, int, int]):
        if mfma == (16, 16, 1, 4):
            return self.v_mfma_f32_16x16x4f32
        assert False

    def materialize(self):
        return "\n".join([inst[0]() for inst in self.instructions])


def gpu_function(func):
    def wrapper(context, *args, **kwargs):
        if context is None:
            context = GpuContext()
        return func(context, *args, **kwargs)

    return wrapper


class DataType(IntEnum):
    FP32 = 0


def datatype_size(dtype: DataType):
    if dtype == DataType.FP32:
        return 4

    assert False, "unrecognized type"

class GemmOptimizations:
    def __init__(self, level: int):
        self.level = level
        self.wgm = 1
        self.plr = 0
        self._setup_optimizations()
        
    def _setup_optimizations(self):
        if self.level != 0:
            self.plr = 1
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
        self.name = None

        if self.lds_usage_bytes >= MAX_LDS_NUM_BYTES:
            raise RuntimeError(f"LDS usage exceeds {MAX_LDS_NUM_BYTES}: {self.lds_usage_bytes}")

        assert all((wave_group[i] & (wave_group[i] - 1)) == 0 for i in range(len(wave_group)))

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

    @property
    def num_unrolled_iters(self) -> int:
        return self.depth_k // self.mfma[3]

    def to_dict(self) -> Dict:
        return {
            "a_type": int(self.a_type),
            "b_type": int(self.b_type),
            "cd_type": int(self.cd_type),
            "scalar_type": int(self.scalar_type),
            "wave_group": self.wave_group,
            "trans_a": self.trans_a,
            "trans_b": self.trans_b,
            "mfma": self.mfma,
            "wave_tiling": self.wave_tiling,
            "depth_k": self.depth_k,
            "wavefront_size": self.wavefront_size,
            "name": self.name if self.name else ""
        }

    def from_dict(self, d):
        self.a_type = DataType(d["a_type"])
        self.b_type = DataType(d["b_type"])
        self.cd_type = DataType(d["cd_type"])
        self.scalar_type = DataType(d["scalar_type"])
        self.wave_group = d["wave_group"]
        self.trans_a = d["trans_a"]
        self.trans_b = d["trans_b"]
        self.mfma = d["mfma"]
        self.wave_tiling = d["wave_tiling"]
        self.depth_k = d["depth_k"]
        self.wavefront_size = d["wavefront_size"]
        self.name = d["name"]

@gpu_function
def gemm(
    context: GpuContext,
    name: str,
    arch: str,
    config: GemmSolutionConfig,
    opt: GemmOptimizations,
    arguments: FunctionArgumentList,
) -> str:
    meta = FunctionMeta(name, arguments)
    meta.group_segment_fixed_size = config.lds_usage_bytes
    meta.workgroup_size = config.wave_group[0] * config.wave_group[1] * meta.wavefront_size

    @dataclass
    class SgprAlloc:
        srd_a: int
        srd_b: int
        srd_c: int
        srd_d: int
        kern_args_addr: int
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
        valu_a: List[List[List[int]]]
        valu_b: List[List[List[int]]]
        valu_c: List[List[int]]
        valu_d: List[List[int]]
        valu_acc: List[List[int]]
        w_id: int
        w_row: int
        w_col: int
        wt_id: int

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
            kern_args_addr=0,
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

    def vgpr_alloc(opt: GemmOptimizations):
        # TODO: for non-NN transposes, swap mt0, mt1 if required
        mt0, mt1 = config.tile_size
        depth_k = config.depth_k
        num_workitems = config.num_workitems

        vgpr_counter = 0
        t_id = 0
        vgpr_counter += 1
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
        wt_id = vgpr_counter
        vgpr_counter += 1

        if vgpr_counter % 2:
            vgpr_counter += 1

        mac_vgpr_start = vgpr_counter

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

        valu_a = []
        for _ in range(opt.plr+1):
            valu_a.append(gl_read_data(config.wave_tiling[0], 1, valu_num_vgpr_a))

        valu_b = []
        for _ in range(opt.plr+1):
            valu_b.append(gl_read_data(1, config.wave_tiling[1], valu_num_vgpr_b))

        print("valu{a, b}")
        print(valu_a)
        print(valu_b)

        #release unused vgprs
        vgpr_counter = mac_vgpr_start

        valu_c = gl_read_data(config.wave_tiling[0], config.wave_tiling[1], 4)
        valu_d = gl_read_data(config.wave_tiling[0], config.wave_tiling[1], 4)

        print("valu{c, d}")
        print(valu_c)
        print(valu_d)

        gl_voffset_c = gl_read_data(config.wave_tiling[0], config.wave_tiling[1], 4)
        gw_voffset_d = gl_read_data(config.wave_tiling[0], config.wave_tiling[1], 4)

        valu_acc = gl_read_data(config.wave_tiling[0], config.wave_tiling[1], 4)

        return VgprAlloc(
            t_id=t_id,
            gl_offset_a=gl_voffset_a,
            gl_offset_b=gl_voffset_b,
            gl_offset_c=gl_voffset_c,
            gl_offset_d=gw_voffset_d,
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
            valu_c=valu_c,
            valu_d=valu_d,
            valu_acc=valu_acc,
            w_id=w_id,
            w_row=w_row,
            w_col=w_col,
            wt_id=wt_id
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
                    SgprRange(sgprs.kern_args_addr, 2),
                    kern_arg_sgpr_offset * 4,
                )
                kern_arg_sgpr_offset += 4
                num_sgpr_kernarg -= 4
            elif num_sgpr_kernarg >= 2:
                context.s_load_dwordx2(
                    SgprRange(sgprs.kern_args + kern_arg_sgpr_offset, 2),
                    SgprRange(sgprs.kern_args_addr, 2),
                    kern_arg_sgpr_offset * 4,
                )
                kern_arg_sgpr_offset += 2
                num_sgpr_kernarg -= 2
            else:
                context.s_load_dword(
                    Sgpr(sgprs.kern_args + kern_arg_sgpr_offset),
                    SgprRange(sgprs.kern_args_addr, 2),
                    kern_arg_sgpr_offset * 4,
                )
                kern_arg_sgpr_offset += 1
                num_sgpr_kernarg -= 1
        context.s_waitcnt(lgkmcnt=0)

        # context.label("test_div")
        # with alloc_tmp_sgpr(4) as stmp:
        #     s0, s1, s2, s3 = stmp.split()
        #     context.s_mov_b32(s0, 5)
        #     context.s_mov_b32(s1, 1)
        #     context.s_div_u32(s2, s3, s0, s1)

        if opt.wgm > 1:
            #FIXME: not working correctly
            context.label("wgm_beg")
            assert (opt.wgm & opt.wgm - 1) == 0
            num_workgroups_x, num_workgroups_y = sgprs.kern_args + 17, sgprs.kern_args + 18
            with alloc_tmp_sgpr(4) as stmps:
                stmp0, stmp1, stmp2, stmp3 = stmps.split()
                log_wgm = int(math.log2(opt.wgm))
                #z = x + y * nwg0
                context.s_mul_i32(stmp0, Sgpr(sgprs.wg_id_y), Sgpr(num_workgroups_x))
                context.s_add_i32(stmp0, stmp0, Sgpr(sgprs.wg_id_x))
                #x = (z % wgm) + z / wgm / nwg1 * wgm
                #y = (z / wgm) % n
                context.s_and_b32(Sgpr(sgprs.wg_id_x), stmp0, opt.wgm-1)
                context.s_lshr_b32(stmp1, stmp0, log_wgm)
                context.s_div_u32(stmp3, stmp2, stmp1, Sgpr(num_workgroups_y))
                context.s_mul_i32(stmp3, stmp3, opt.wgm)
                context.s_add_i32(Sgpr(sgprs.wg_id_x), Sgpr(sgprs.wg_id_x), stmp3)
                context.comment("wg_id_x")
                context.s_lshr_b32(Sgpr(sgprs.wg_id_y), stmp0, log_wgm)
                context.s_div_u32(stmp2, stmp1, Sgpr(sgprs.wg_id_y), Sgpr(num_workgroups_y))
                context.comment("wg_id_y")
                context.s_mov_b32(Sgpr(sgprs.wg_id_y), stmp1)
            context.label("wgm_end")

        context.comment("Setup Srd{A, B}")
        context.s_mov_b32(Sgpr(sgprs.srd_a + 3), 0x20000)
        context.s_mov_b32(Sgpr(sgprs.srd_b + 3), 0x20000)
        context.s_mov_b64(SgprRange(sgprs.srd_a, 2), SgprRange(sgprs.kern_args, 2))
        context.s_mov_b64(SgprRange(sgprs.srd_b, 2), SgprRange(sgprs.kern_args + 2, 2))
        context.comment("Setup sizes, m, n and k")
        context.s_mov_b32(Sgpr(sgprs.m), Sgpr(sgprs.kern_args + 8))
        context.s_mov_b32(Sgpr(sgprs.n), Sgpr(sgprs.kern_args + 9))
        context.s_mov_b32(Sgpr(sgprs.k), Sgpr(sgprs.kern_args + 10))
        context.s_mov_b32(Sgpr(sgprs.stride_a_0), 1)
        context.s_mov_b32(Sgpr(sgprs.stride_a_1), Sgpr(sgprs.kern_args + 11))
        context.s_mov_b32(Sgpr(sgprs.stride_b_0), 1)
        context.s_mov_b32(Sgpr(sgprs.stride_b_1), Sgpr(sgprs.kern_args + 12))
        context.s_mov_b32(Sgpr(sgprs.stride_c_0), 1)
        context.s_mov_b32(Sgpr(sgprs.stride_c_1), Sgpr(sgprs.kern_args + 13))
        context.s_mov_b32(Sgpr(sgprs.stride_d_0), 1)
        context.s_mov_b32(Sgpr(sgprs.stride_d_1), Sgpr(sgprs.kern_args + 14))

        context.label("setup_gl_offsets")
        context.comment("Setup global read offsets")
        context.s_lshl_b32(Sgpr(sgprs.row_idx), Sgpr(sgprs.wg_id_x), int(math.log2(config.tile_size[0])))
        context.s_lshl_b32(Sgpr(sgprs.col_idx), Sgpr(sgprs.wg_id_y), int(math.log2(config.tile_size[1])))

        bpe_log_a = int(math.log2(datatype_size(gemm_config.a_type)))
        bpe_log_b = int(math.log2(datatype_size(gemm_config.b_type)))

        with alloc_tmp_sgpr(1) as tmp:
            context.s_mul_i32(tmp, Sgpr(sgprs.stride_a_1), Sgpr(sgprs.k))
            context.s_lshl_b32(Sgpr(sgprs.srd_a + 2), tmp, bpe_log_a)
            context.s_mul_i32(tmp, Sgpr(sgprs.n), Sgpr(sgprs.stride_b_1))
            context.s_lshl_b32(Sgpr(sgprs.srd_b + 2), tmp, bpe_log_b)

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

        vgprs = vgpr_alloc(opt)
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
        context.v_lshrrev_b32(
            Vgpr(vgprs.t_col), int(math.log2(num_load_threads0_a)), Vgpr(vgprs.t_id)
        )

        for j, col in enumerate(vgprs.gl_offset_a):
            for i, row in enumerate(col):
                with alloc_tmp_sgpr(1) as stmp:
                    context.comment(f"gl_addr_a_{i}_{j}")
                    context.s_mov_b32(stmp, j * num_load_threads1_a)
                    context.v_add_u32(Vgpr(vgprs.gl_offset_a[j][i]), Vgpr(vgprs.t_col), stmp)
                    context.v_mul_lo_u32(
                        Vgpr(vgprs.gl_offset_a[j][i]),
                        Vgpr(vgprs.gl_offset_a[j][i]),
                        Sgpr(sgprs.stride_a_1),
                    )
                    context.v_add_u32(
                        Vgpr(vgprs.gl_offset_a[j][i]),
                        Vgpr(vgprs.gl_offset_a[j][i]),
                        Vgpr(vgprs.t_row),
                    )
                    context.s_mov_b32(stmp, i * num_load_threads0_a * gl_num_elements_a)
                    context.v_add_u32(Vgpr(vgprs.gl_offset_a[j][i]), Vgpr(vgprs.gl_offset_a[j][i]), stmp)
                    context.v_mul_lo_u32(
                        Vgpr(vgprs.gl_offset_a[j][i]),
                        datatype_size(config.a_type),
                        Vgpr(vgprs.gl_offset_a[j][i]),
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
        context.v_lshrrev_b32(
            Vgpr(vgprs.t_col), int(math.log2(num_load_threads0_b)), Vgpr(vgprs.t_id)
        )

        for j, col in enumerate(vgprs.gl_offset_b):
            for i, row in enumerate(col):
                with alloc_tmp_sgpr(1) as stmp:
                    context.comment(f"gl_addr_b_{i}_{j}")
                    context.s_mov_b32(stmp, j * num_load_threads1_b)
                    context.v_add_u32(Vgpr(vgprs.gl_offset_b[j][i]), Vgpr(vgprs.t_col), stmp)
                    context.v_mul_lo_u32(
                        Vgpr(vgprs.gl_offset_b[j][i]),
                        Vgpr(vgprs.gl_offset_b[j][i]),
                        Sgpr(sgprs.stride_b_1),
                    )
                    context.v_add_u32(
                        Vgpr(vgprs.gl_offset_b[j][i]),
                        Vgpr(vgprs.gl_offset_b[j][i]),
                        Vgpr(vgprs.t_row),
                    )
                    context.s_mov_b32(stmp, i * num_load_threads0_b * gl_num_elements_b)
                    context.v_add_u32(Vgpr(vgprs.gl_offset_b[j][i]), Vgpr(vgprs.gl_offset_b[j][i]), stmp)
                    context.v_mul_lo_u32(
                        Vgpr(vgprs.gl_offset_b[j][i]),
                        Vgpr(vgprs.gl_offset_b[j][i]),
                        datatype_size(config.b_type),
                    )

        def gl_a():
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

        def gl_b():
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

        def gl_increments():
            with alloc_tmp_sgpr(1) as stmp:
                context.comment("gl_offset increments for unrolled loop")
                context.s_mul_i32(stmp, Sgpr(sgprs.stride_a_1), config.depth_k * datatype_size(config.a_type))
                context.s_add_i32(Sgpr(sgprs.gl_offset_a), Sgpr(sgprs.gl_offset_a), stmp)
                context.s_add_i32(Sgpr(sgprs.gl_offset_b), Sgpr(sgprs.gl_offset_b), config.depth_k * datatype_size(config.b_type))

        gl_a()
        gl_b()

        context.comment("lw_a")
        context.v_and_b32(
            Vgpr(vgprs.t_row),
            Vgpr(vgprs.t_id),
            config.tile_size[0] // gl_num_elements_a - 1,
        )

        context.v_mul_lo_u32(Vgpr(vgprs.t_row), Vgpr(vgprs.t_row), gl_num_elements_a)
        context.v_lshrrev_b32(
            Vgpr(vgprs.t_col), int(math.log2(num_load_threads0_a)), Vgpr(vgprs.t_id)
        )

        for j, col in enumerate(vgprs.lw_addr_a):
            for i, row in enumerate(col):
                with alloc_tmp_sgpr(1) as stmp:
                    context.comment(f"lw_addr_a_{i}_{j}")
                    context.s_mov_b32(stmp, j * num_load_threads1_a)
                    context.v_add_u32(Vgpr(row), Vgpr(vgprs.t_col), stmp)
                    context.s_mov_b32(stmp, config.tile_size[0])
                    context.v_mul_lo_u32(Vgpr(row), Vgpr(row), stmp)
                    context.s_mov_b32(stmp, i * num_load_threads0_a * gl_num_elements_a)
                    context.v_add_u32(Vgpr(row), Vgpr(row), stmp)
                context.v_add_u32(Vgpr(row), Vgpr(row), Vgpr(vgprs.t_row))
                context.v_mul_lo_u32(Vgpr(row), Vgpr(row), datatype_size(config.a_type))

        context.comment("lw_b")
        context.v_and_b32(
            Vgpr(vgprs.t_row), Vgpr(vgprs.t_id), config.depth_k // gl_num_elements_b - 1
        )
        context.v_mul_lo_u32(Vgpr(vgprs.t_row), Vgpr(vgprs.t_row), gl_num_elements_b)
        context.v_lshrrev_b32(
            Vgpr(vgprs.t_col), int(math.log2(num_load_threads0_b)), Vgpr(vgprs.t_id)
        )

        for j, col in enumerate(vgprs.lw_addr_b):
            for i, row in enumerate(col):
                with alloc_tmp_sgpr(1) as stmp:
                    context.comment(f"lw_addr_b_{i}_{j}")
                    context.s_mov_b32(stmp, j * num_load_threads1_b)
                    context.v_add_u32(Vgpr(row), Vgpr(vgprs.t_col), stmp)
                    context.s_mov_b32(stmp, config.depth_k)
                    context.v_mul_lo_u32(Vgpr(row), Vgpr(row), stmp)
                    context.s_mov_b32(stmp, i * num_load_threads0_b * gl_num_elements_b)
                    context.v_add_u32(Vgpr(row), Vgpr(row), stmp)
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

        def gl_increments_swap_lds():
            gl_increments()

            context.comment("swap ds write address")
            context.s_mov_b32(Sgpr(sgprs.lds_start_addr), config.lds_swap_offset_bytes)
            for j, col in enumerate(vgprs.lw_addr_a):
                for i, row in enumerate(col):
                    context.v_add_u32(Vgpr(row), Vgpr(row), Sgpr(sgprs.lds_start_addr))

            for j, col in enumerate(vgprs.lw_addr_b):
                for i, row in enumerate(col):
                    context.v_add_u32(Vgpr(row), Vgpr(row), Sgpr(sgprs.lds_start_addr))

        gl_increments_swap_lds()
        context.label("lds_wave_offsets")
        context.comment("lds read addresses: wave offsets")
        context.v_lshrrev_b32(Vgpr(vgprs.w_id), 6, Vgpr(vgprs.t_id))
        context.v_and_b32(Vgpr(vgprs.wt_id), config.wavefront_size-1, Vgpr(vgprs.t_id))
        context.v_lshrrev_b32(Vgpr(vgprs.w_col), int(math.log2(config.wave_group[0])), Vgpr(vgprs.w_id))
        context.v_mul_lo_u32(Vgpr(vgprs.w_col), Vgpr(vgprs.w_col), config.mfma[1])
        context.v_and_b32(Vgpr(vgprs.w_row), config.wave_group[0]-1, Vgpr(vgprs.w_id))
        context.v_mul_lo_u32(Vgpr(vgprs.w_row), Vgpr(vgprs.w_row), config.mfma[0])

        context.comment("lds read addresses: thread offsets a")
        context.v_and_b32(Vgpr(vgprs.t_row), config.mfma[0]-1, Vgpr(vgprs.wt_id))
        context.v_lshrrev_b32(Vgpr(vgprs.t_col), int(math.log2(config.mfma[0])), Vgpr(vgprs.wt_id))
        #TODO: multiply num_reg_per_thread_mfma_a for other MFMA
        context.v_add_u32(Vgpr(vgprs.t_row), Vgpr(vgprs.t_row), Vgpr(vgprs.w_row))

        for j, col in enumerate(vgprs.lr_addr_a):
            for i, row in enumerate(col):
                with alloc_tmp_sgpr(1) as stmp:
                    context.s_mov_b32(stmp, config.tile_size[0])
                    context.v_mul_lo_u32(Vgpr(row), Vgpr(vgprs.t_col), stmp)
                    context.v_add_u32(Vgpr(row), Vgpr(row), Vgpr(vgprs.t_row))
                    context.v_mul_lo_u32(Vgpr(row), Vgpr(row), datatype_size(config.a_type))
                    context.s_mov_b32(stmp, config.wave_group[0] * config.mfma[0])
                    context.v_add_u32(Vgpr(vgprs.t_row), Vgpr(vgprs.t_row), stmp)

        context.comment("lds read addresses: thread offsets b")
        context.v_and_b32(Vgpr(vgprs.t_col), config.mfma[1]-1, Vgpr(vgprs.wt_id))
        context.v_lshrrev_b32(Vgpr(vgprs.t_row), int(math.log2(config.mfma[1])), Vgpr(vgprs.wt_id))
        #TODO: multiply num_reg_per_thread_mfma_b for other MFMA
        context.v_add_u32(Vgpr(vgprs.t_col), Vgpr(vgprs.t_col), Vgpr(vgprs.w_col))

        for j, col in enumerate(vgprs.lr_addr_b):
            for i, row in enumerate(col):
                with alloc_tmp_sgpr(1) as stmp:
                    context.s_mov_b32(stmp, config.depth_k)
                    context.v_mul_lo_u32(Vgpr(row), Vgpr(vgprs.t_col), stmp)
                    context.v_add_u32(Vgpr(row), Vgpr(row), Vgpr(vgprs.t_row))
                    context.v_mul_lo_u32(Vgpr(row), Vgpr(row), datatype_size(config.b_type))
                    context.s_mov_b32(stmp, config.wave_group[1] * config.mfma[1])
                    context.v_add_u32(Vgpr(vgprs.t_col), Vgpr(vgprs.t_col), stmp)

        context.comment("sync prefetch")
        context.s_waitcnt(lgkmcnt=0)
        context.s_barrier()

        unrolled_lr_offset_a, unrolled_lr_offset_b = config.lds_offset_bytes

        def lr_a(k: int):
            nonlocal unrolled_lr_offset_a
            for j, col in enumerate(vgprs.valu_a[k]):
                for i, row in enumerate(col):
                    context.ds_read_inst(config.num_bytes_per_ds_read[0])(Vgpr(row), Vgpr(vgprs.lr_addr_a[j][i]), unrolled_lr_offset_a)
            unrolled_lr_offset_a += config.mfma[3] * config.tile_size[0] * datatype_size(config.a_type)

        def lr_b(k: int):
            nonlocal unrolled_lr_offset_b
            for j, col in enumerate(vgprs.valu_b[k]):
                for i, row in enumerate(col):
                    context.ds_read_inst(config.num_bytes_per_ds_read[1])(Vgpr(row), Vgpr(vgprs.lr_addr_b[j][i]), unrolled_lr_offset_b)
            unrolled_lr_offset_b += config.mfma[3] * datatype_size(config.b_type)

        context.label("outer_loop")

        plr_buf_idx = 0

        for u in range(opt.plr):
            lr_a(plr_buf_idx)
            lr_b(plr_buf_idx)
            plr_buf_idx = (plr_buf_idx + 1) % (opt.plr + 1)

        gl_a()
        gl_b()

        def mfma(k: int):
            for j, col in enumerate(agprs.arpgs):
                for i, row in enumerate(col):
                    context.mfma_inst(config.mfma)(AccVgprRange(row, 4), Vgpr(vgprs.valu_a[k][0][i]), Vgpr(vgprs.valu_b[k][j][0]), AccVgprRange(row, 4))

        def lr_a_gen(k: int):
            nonlocal unrolled_lr_offset_a
            for j, col in enumerate(vgprs.valu_a[k]):
                for i, row in enumerate(col):
                    yield lambda: context.ds_read_inst(config.num_bytes_per_ds_read[0])(Vgpr(row), Vgpr(vgprs.lr_addr_a[j][i]), unrolled_lr_offset_a)
            unrolled_lr_offset_a += config.mfma[3] * config.tile_size[0] * datatype_size(config.a_type)

        def lr_b_gen(k: int):
            nonlocal unrolled_lr_offset_b
            for j, col in enumerate(vgprs.valu_b[k]):
                for i, row in enumerate(col):
                    yield context.ds_read_inst(config.num_bytes_per_ds_read[1])(Vgpr(row), Vgpr(vgprs.lr_addr_b[j][i]), unrolled_lr_offset_b)
            unrolled_lr_offset_b += config.mfma[3] * datatype_size(config.b_type)

        def mfma_gen(k):
            for j, col in enumerate(agprs.arpgs):
                for i, row in enumerate(col):
                    yield lambda: context.mfma_inst(config.mfma)(AccVgprRange(row, 4), Vgpr(vgprs.valu_a[k][0][i]), Vgpr(vgprs.valu_b[k][j][0]), AccVgprRange(row, 4))

        def lw_a():
            for j, col in enumerate(vgprs.lw_addr_a):
                for i, row in enumerate(col):
                    vdata = (
                        VgprRange(
                            vgprs.gl_data_a[j][i], config.num_bytes_per_buffer_load[0] // 4
                        )
                        if config.num_bytes_per_buffer_load[0] > 4
                            else Vgpr(vgprs.gl_data_a[j][i])
                        )
                    context.ds_write_inst(config.num_bytes_per_buffer_load[0])(Vgpr(row), vdata, config.lds_offset_bytes[0])

        def lw_b():
            for j, col in enumerate(vgprs.lw_addr_b):
                for i, row in enumerate(col):
                    vdata = (
                        VgprRange(
                            vgprs.gl_data_b[j][i], config.num_bytes_per_buffer_load[1] // 4
                        )
                        if config.num_bytes_per_buffer_load[1] > 4
                            else Vgpr(vgprs.gl_data_b[j][i])
                        )
                    context.ds_write_inst(config.num_bytes_per_buffer_load[1])(Vgpr(row), vdata, config.lds_offset_bytes[1])

        def swap_lds_addr():
            for j, col in enumerate(vgprs.lr_addr_a):
                for i, row in enumerate(col):
                    context.v_add_i32(Vgpr(row), Vgpr(row), Sgpr(sgprs.lds_start_addr))

            for j, col in enumerate(vgprs.lr_addr_b):
                for i, row in enumerate(col):
                    context.v_add_i32(Vgpr(row), Vgpr(row), Sgpr(sgprs.lds_start_addr))

            context.s_mul_i32(Sgpr(sgprs.lds_start_addr), Sgpr(sgprs.lds_start_addr), -1)

            for j, col in enumerate(vgprs.lw_addr_a):
                for i, row in enumerate(col):
                    context.v_add_i32(Vgpr(row), Vgpr(row), Sgpr(sgprs.lds_start_addr))

            for j, col in enumerate(vgprs.lw_addr_b):
                for i, row in enumerate(col):
                    context.v_add_i32(Vgpr(row), Vgpr(row), Sgpr(sgprs.lds_start_addr))

        if opt.level:
            for u in range(config.num_unrolled_iters):
                context.s_waitcnt(lgkmcnt=0)
                next_plr_buf_idx = (plr_buf_idx + 1) % (opt.plr + 1)
                mfma_iter = mfma_gen(u%(opt.plr+1))
                if u + opt.plr < config.num_unrolled_iters:
                    for inst in roundrobin(mfma_iter, lr_a_gen(plr_buf_idx), mfma_iter, lr_b_gen(plr_buf_idx)):
                        if inst:
                            inst()
                else:
                    mfma(u%(opt.plr+1))
                plr_buf_idx = next_plr_buf_idx
        elif opt.plr:
            for u in range(config.num_unrolled_iters):
                context.s_waitcnt(lgkmcnt=0)
                mfma(u%(opt.plr+1))
                next_plr_buf_idx = (plr_buf_idx + 1) % (opt.plr + 1)
                if u + opt.plr < config.num_unrolled_iters:
                    lr_a(plr_buf_idx)
                    lr_b(plr_buf_idx)
                plr_buf_idx = next_plr_buf_idx
        else:
            for u in range(config.num_unrolled_iters):
                lr_a(plr_buf_idx)
                lr_b(plr_buf_idx)
                context.s_waitcnt(lgkmcnt=0)
                mfma(u%(opt.plr+1))
                next_plr_buf_idx = (plr_buf_idx + 1) % (opt.plr + 1)
                plr_buf_idx = next_plr_buf_idx

        context.comment("ds write")
        context.s_waitcnt(vmcnt=0)
        lw_a()
        lw_b()
        context.comment("swap lds")
        swap_lds_addr()
        context.comment("wait for ds writes")
        context.s_waitcnt(lgkmcnt=0)
        context.s_barrier()

        with alloc_tmp_sgpr(1) as stmp:
            context.s_add_i32(Sgpr(sgprs.k_idx), Sgpr(sgprs.k_idx), config.depth_k)
            context.s_mul_i32(stmp, Sgpr(sgprs.stride_a_1), config.depth_k * datatype_size(config.a_type))
            context.s_add_i32(Sgpr(sgprs.gl_offset_a), Sgpr(sgprs.gl_offset_a), stmp)
            context.s_add_i32(Sgpr(sgprs.gl_offset_b), Sgpr(sgprs.gl_offset_b), config.depth_k * datatype_size(config.a_type))
            context.s_add_i32(stmp, Sgpr(sgprs.k_idx), config.depth_k)
            context.s_cmp_lt_u32(stmp, Sgpr(sgprs.k))
            context.s_cbranch_scc1("outer_loop")
            context.label("prefetch_last_loop")
            context.comment("prefetch last loop")

        unrolled_lr_offset_a, unrolled_lr_offset_b = config.lds_offset_bytes

        plr_buf_idx = 0

        for u in range(opt.plr):
            lr_a(plr_buf_idx)
            lr_b(plr_buf_idx)
            plr_buf_idx = (plr_buf_idx + 1) % (opt.plr + 1)

        if opt.level:
            for u in range(config.num_unrolled_iters):
                context.s_waitcnt(lgkmcnt=0)
                next_plr_buf_idx = (plr_buf_idx + 1) % (opt.plr + 1)
                if u + opt.plr < config.num_unrolled_iters:
                    mfma_iter = mfma_gen(u%(opt.plr+1))
                    for inst in roundrobin(mfma_iter, lr_a_gen(plr_buf_idx), mfma_iter, lr_b_gen(plr_buf_idx)):
                        if inst:
                            inst()
                else:
                    mfma(u%(opt.plr+1))
                plr_buf_idx = next_plr_buf_idx
        elif opt.plr:
            for u in range(config.num_unrolled_iters):
                context.s_waitcnt(lgkmcnt=0)
                mfma(u%(opt.plr+1))
                next_plr_buf_idx = (plr_buf_idx + 1) % (opt.plr + 1)
                if u + opt.plr < config.num_unrolled_iters:
                    lr_a(plr_buf_idx)
                    lr_b(plr_buf_idx)
                plr_buf_idx = next_plr_buf_idx
        else:
            for u in range(config.num_unrolled_iters):
                lr_a(plr_buf_idx)
                lr_b(plr_buf_idx)
                context.s_waitcnt(lgkmcnt=0)
                mfma(u%(opt.plr+1))
                next_plr_buf_idx = (plr_buf_idx + 1) % (opt.plr + 1)
                plr_buf_idx = next_plr_buf_idx

        context.comment("setup srd{c, d}")
        context.s_mov_b64(SgprRange(sgprs.srd_c, 2), SgprRange(sgprs.kern_args+4, 2))
        context.s_mov_b64(SgprRange(sgprs.srd_d, 2), SgprRange(sgprs.kern_args+6, 2))
        context.s_mov_b32(Sgpr(sgprs.srd_c+3), 0x20000)
        context.s_mov_b32(Sgpr(sgprs.srd_d+3), 0x20000)
        context.s_mov_b32(Sgpr(sgprs.srd_c+2), Sgpr(sgprs.m))
        context.s_mul_i32(Sgpr(sgprs.srd_c+2), Sgpr(sgprs.srd_c+2), Sgpr(sgprs.stride_c_1))
        context.s_mul_i32(Sgpr(sgprs.srd_c+2), Sgpr(sgprs.srd_c+2), datatype_size(config.cd_type))
        context.s_mov_b32(Sgpr(sgprs.srd_d+2), Sgpr(sgprs.m))
        context.s_mul_i32(Sgpr(sgprs.srd_d+2), Sgpr(sgprs.srd_d+2), Sgpr(sgprs.stride_d_1))
        context.s_mul_i32(Sgpr(sgprs.srd_d+2), Sgpr(sgprs.srd_d+2), datatype_size(config.cd_type))
        #re-use
        gl_offset_c = sgprs.gl_offset_a
        gw_offset_d = sgprs.gl_offset_b
        context.s_mul_i32(Sgpr(gl_offset_c), Sgpr(sgprs.col_idx), Sgpr(sgprs.stride_c_1))
        context.s_add_i32(Sgpr(gl_offset_c), Sgpr(gl_offset_c), Sgpr(sgprs.row_idx))
        context.s_mul_i32(Sgpr(gl_offset_c), Sgpr(gl_offset_c), datatype_size(config.cd_type))
        context.s_mul_i32(Sgpr(gw_offset_d), Sgpr(sgprs.col_idx), Sgpr(sgprs.stride_d_1))
        context.s_add_i32(Sgpr(gw_offset_d), Sgpr(gw_offset_d), Sgpr(sgprs.row_idx))
        context.s_mul_i32(Sgpr(gw_offset_d), Sgpr(gw_offset_d), datatype_size(config.cd_type))

        for j, col in enumerate(vgprs.gl_offset_d):
            for i, row in enumerate(col):
                context.comment(f"gw_addr_{i}_{j}")
                with alloc_tmp_sgpr(1) as stmp:
                    context.v_and_b32(Vgpr(vgprs.t_col), config.mfma[1]-1, Vgpr(vgprs.wt_id))
                    context.s_mov_b32(stmp,  j * config.wave_group[1] * config.mfma[1])
                    context.v_add_u32(Vgpr(vgprs.t_col), Vgpr(vgprs.t_col), stmp)
                    context.v_lshrrev_b32(Vgpr(vgprs.t_row), int(math.log2(config.mfma[1])), Vgpr(vgprs.wt_id))
                    context.v_mul_lo_u32(Vgpr(vgprs.t_row), 4, Vgpr(vgprs.t_row))
                    context.s_mov_b32(stmp, i * config.wave_group[0] * config.mfma[0])
                    context.v_add_u32(Vgpr(vgprs.t_row), Vgpr(vgprs.t_row), stmp)
                    context.v_add_i32(Vgpr(vgprs.t_col), Vgpr(vgprs.t_col), Vgpr(vgprs.w_col))
                    context.v_add_i32(Vgpr(vgprs.t_row), Vgpr(vgprs.t_row), Vgpr(vgprs.w_row))
                    context.comment(f"setup voffset_c_{i}_{j}")
                    context.v_mul_lo_u32(Vgpr(vgprs.gl_offset_c[j][i]), Vgpr(vgprs.t_col), Sgpr(sgprs.stride_c_1))
                    context.v_add_u32(Vgpr(vgprs.gl_offset_c[j][i]), Vgpr(vgprs.gl_offset_c[j][i]), Vgpr(vgprs.t_row))
                    context.v_mul_lo_u32(Vgpr(vgprs.gl_offset_c[j][i]), Vgpr(vgprs.gl_offset_c[j][i]), datatype_size(config.cd_type))
                    context.comment(f"setup voffset_d_{i}_{j}")
                    context.v_mul_lo_u32(Vgpr(vgprs.gl_offset_d[j][i]), Vgpr(vgprs.t_col), Sgpr(sgprs.stride_d_1))
                    context.v_add_u32(Vgpr(vgprs.gl_offset_d[j][i]), Vgpr(vgprs.gl_offset_d[j][i]), Vgpr(vgprs.t_row))
                    context.v_mul_lo_u32(Vgpr(vgprs.gl_offset_d[j][i]), Vgpr(vgprs.gl_offset_d[j][i]), datatype_size(config.cd_type))

        context.s_mov_b32(Sgpr(sgprs.alpha), Sgpr(sgprs.kern_args+15))
        context.s_mov_b32(Sgpr(sgprs.beta), Sgpr(sgprs.kern_args+16))

        context.label("gw")
        for j, col in enumerate(agprs.arpgs):
            for i, row in enumerate(col):
                context.comment(f"gw_{i}_{j}")
                for r in range(agprs.num_reg_per_thread):
                    context.v_accvgpr_read_b32(Vgpr(vgprs.valu_acc[j][i]+r), AccVgpr(row+r))
                context.buffer_load_inst(agprs.num_reg_per_thread)(VgprRange(vgprs.valu_c[j][i], agprs.num_reg_per_thread), Vgpr(vgprs.gl_offset_c[j][i]), SgprRange(sgprs.srd_c, 4), Sgpr(gl_offset_c), 0)
                for r in range(agprs.num_reg_per_thread):
                    context.v_mul_f32(Vgpr(vgprs.valu_acc[j][i]+r), Vgpr(vgprs.valu_acc[j][i]+r), Sgpr(sgprs.alpha))
                context.s_waitcnt(vmcnt=0)
                for r in range(agprs.num_reg_per_thread):
                    context.v_fma_f32(Vgpr(vgprs.valu_acc[j][i]+r), Sgpr(sgprs.beta), Vgpr(vgprs.valu_c[j][i]+r), Vgpr(vgprs.valu_acc[j][i]+r))
                context.buffer_store_inst(agprs.num_reg_per_thread)(VgprRange(vgprs.valu_acc[j][i], agprs.num_reg_per_thread), Vgpr(vgprs.gl_offset_d[j][i]), SgprRange(sgprs.srd_d, 4), Sgpr(gw_offset_d), 0)

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
    (4, 2),
    16,
    False,
    False,
)
print(gemm_config.tile_size, gemm_config.num_workitems)
print(gemm_config.num_bytes_per_buffer_load)

arch = "gfx90a:xnack-"
opt = GemmOptimizations(1)
opt.plr = 1
asm_str = gemm(
    None,
    "gemm",
    arch,
    gemm_config,
    opt,
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
        FunctionArgument("by_value", "numWorkgroupX", None, 4),
        FunctionArgument("by_value", "numWorkgroupY", None, 4),
    ],
)

with open('generated_gemm.s', 'w') as f:
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
            "generated_gemm.o",
        ]
    )
    ret = subprocess.run(
        [DEFAULT_CLANG_PATH, "-target", "amdgcn-amd-amdhsa", "generated_gemm.o", "-o", "generated_gemm.co"]
    )

with open("generated_gemm.toml", "wb") as f:
    tomli_w.dump(gemm_config.to_dict(), f)
