.amdgcn_target "amdgcn-amd-amdhsa--gfx90a:xnack-"
.text
.globl set_func
.p2align 8
.type set_func,@function
set_func:
  s_load_dwordx2 s[0:1], s[0:1] 0x0
  s_mov_b32 s[2], 0x4
  s_mov_b32 s[3], 0x20000
  v_mov_b32 v1, 55.66
  v_mov_b32 v0, 0x0
  s_waitcnt lgkmcnt(0)
  buffer_store_dword v1, v0, s[0:3], 0 offen offset:0
  s_endpgm
.Lset_func_end0:
  .size set_func, .Lset_func_end0 - set_func

.rodata
.p2align 6
.amdhsa_kernel set_func
  .amdhsa_user_sgpr_kernarg_segment_ptr 1
  .amdhsa_accum_offset 4
  .amdhsa_next_free_vgpr .amdgcn.next_free_vgpr
  .amdhsa_next_free_sgpr .amdgcn.next_free_sgpr
.end_amdhsa_kernel

.amdgpu_metadata
---
amdhsa.version:
 - 1
 - 1

amdhsa.kernels:
 - .name: set_func
   .symbol: set_func.kd
   .kernarg_segment_size: 8
   .group_segment_fixed_size: 0
   .private_segment_fixed_size: 0
   .kernarg_segment_align: 8
   .wavefront_size: 64
   .sgpr_count: 6
   .vgpr_count: 2
   .agpr_count: 0
   .max_flat_workgroup_size: 256
   .args:
     - .size: 8
       .offset: 0
       .value_kind: global_buffer
       .address_space: global
       .actual_access: read_write
.end_amdgpu_metadata