.amdgcn_target "amdgcn-amd-amdhsa--gfx90a:xnack-"
.text
.globl max_func
.p2align 8
.type max_func,@function
max_func:
  .set sgprKernelArg, 0
  .set sgprWorkgroupId, 2
  .set sgprWorkgroupOffset, 2
  .set sgprNumElemInBytes, 3
  .set sgprSrdInput, 4
  .set sgprSrdOutput, 4
  .set sgprReduceStep, 8
  .set sgprNumReduces, 9
  .set sgprReduceMask, 10
  .set sgprReduceIter, 11
  .set sgprOutputWorkgroupOffset, 12
  .set srdConst, 0x20000
  .set workgroupSize, 256
  .set log2WorkgroupSize, 8
  .set log2ElemWidthByte, 2
  .set elemWidthByte, 4
  .set inputOffset, 0x0
  .set inputSizeOffset, 0x8
  .set outputOffset, 0xc
  .set vgprTid, 0
  .set vgprElemOffsetByte, 1
  .set vgprElem, 2
  .set vgprMaxElem, 3
  .set vgprDSWriteOffsetByte, 4
  .set vgprReduceTemp, 5
  .set vgprTidRem, 7
  .set vgprReduceByteOffset, 8
  s_load_dwordx2 s[sgprSrdInput:sgprSrdInput+1], s[sgprKernelArg:sgprKernelArg+1] inputOffset
  s_load_dword s[sgprNumElemInBytes], s[sgprKernelArg:sgprKernelArg+1] inputSizeOffset
  s_waitcnt lgkmcnt(0)
  s_mov_b32 s[sgprNumReduces], workgroupSize
  s_lshl_b32 s[sgprNumElemInBytes], s[sgprNumElemInBytes], log2ElemWidthByte
  s_mov_b32 s[sgprSrdInput+2], s[sgprNumElemInBytes]
  s_mov_b32 s[sgprSrdInput+3], srdConst
  s_lshl_b32 s[sgprOutputWorkgroupOffset], s[sgprWorkgroupId], log2ElemWidthByte
  s_lshl_b32 s[sgprWorkgroupOffset], s[sgprWorkgroupId], log2WorkgroupSize
  v_add_lshl_u32 v[vgprElemOffsetByte], v[vgprTid], s[sgprWorkgroupOffset], log2ElemWidthByte
  buffer_load_dword v[vgprElem], v[vgprElemOffsetByte], s[sgprSrdInput:sgprSrdInput+3], 0 offen offset:0
  s_waitcnt vmcnt(0)
  v_lshlrev_b32 v[vgprDSWriteOffsetByte], log2ElemWidthByte, v[vgprTid]
  ds_write_b32 v[vgprDSWriteOffsetByte], v[vgprElem]
  s_mov_b32 s[sgprReduceIter], 0
  s_waitcnt lgkmcnt(0)
  s_barrier
  label_reduce:
    s_lshl_b32 s[sgprReduceStep], 1, s[sgprReduceIter]
    s_cmp_lt_u32 s[sgprReduceStep], s[sgprNumReduces]
    s_cbranch_scc0 label_reduceend
    s_lshl_b32 s[sgprReduceMask], s[sgprReduceStep], 1
    s_sub_i32 s[sgprReduceMask], s[sgprReduceMask], 1
    v_and_b32 v[vgprTidRem], v[vgprTid], s[sgprReduceMask]
    v_cmpx_eq_i32 vcc, 0, v[vgprTidRem]
    v_lshlrev_b32 v[vgprReduceByteOffset], log2ElemWidthByte, v[vgprTid]
    v_add_lshl_u32 v[vgprReduceByteOffset+1], v[vgprTid], s[sgprReduceStep], log2ElemWidthByte
    ds_read_b32 v[vgprReduceTemp], v[vgprReduceByteOffset]
    ds_read_b32 v[vgprReduceTemp+1], v[vgprReduceByteOffset+1]
    s_waitcnt lgkmcnt(0)
    v_max_f32 v[vgprMaxElem], v[vgprReduceTemp], v[vgprReduceTemp+1]
    ds_write_b32 v[vgprReduceByteOffset], v[vgprMaxElem]
    s_add_i32 s[sgprReduceIter], s[sgprReduceIter], 1
    s_waitcnt lgkmcnt(0)
    s_barrier
    s_branch label_reduce
  label_reduceend:
    s_load_dwordx2 s[sgprSrdOutput:sgprSrdOutput+1], s[sgprKernelArg:sgprKernelArg+1] outputOffset
    s_waitcnt lgkmcnt(0)
    s_add_u32 s[sgprSrdOutput], s[sgprSrdOutput], s[sgprOutputWorkgroupOffset]
    buffer_store_dword v[vgprMaxElem], v[vgprTid], s[sgprSrdOutput:sgprSrdOutput+3], 0 offen offset:0
  s_endpgm
.Lmax_func_end0:
  .size max_func, .Lmax_func_end0 - max_func

.rodata
.p2align 6
.amdhsa_kernel max_func
  .amdhsa_user_sgpr_kernarg_segment_ptr 1
  .amdhsa_system_sgpr_workgroup_id_x 1
  .amdhsa_accum_offset 12
  .amdhsa_next_free_vgpr .amdgcn.next_free_vgpr
  .amdhsa_next_free_sgpr .amdgcn.next_free_sgpr
.end_amdhsa_kernel

.amdgpu_metadata
---
amdhsa.version:
 - 1
 - 1

amdhsa.kernels:
 - .name: max_func
   .symbol: max_func.kd
   .kernarg_segment_size: 24
   .group_segment_fixed_size: 0
   .private_segment_fixed_size: 0
   .kernarg_segment_align: 8
   .wavefront_size: 64
   .sgpr_count: 13
   .vgpr_count: 16
   .agpr_count: 0
   .max_flat_workgroup_size: 256
   .args:
     - .size: 8
       .offset: 0
       .value_kind: global_buffer
       .address_space: global
       .actual_access: read_write
     - .size: 4
       .offset: 8
       .value_kind: by_value
     - .size: 8
       .offset: 12
       .value_kind: global_buffer
       .address_space: global
       .actual_access: read_write
.end_amdgpu_metadata