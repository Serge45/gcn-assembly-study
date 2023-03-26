.amdgcn_target "amdgcn-amd-amdhsa--gfx90a:xnack-"
.text
.globl relu
.p2align 8
.type relu,@function
relu:
  s_load_dwordx2 s[4:5], s[0:1] 0    //s[4:7] for Srd a
  s_load_dwordx2 s[8:9], s[0:1] 8    //s[8:11] for Srd b
  s_load_dword s[0], s[0:1] 16       //reuse s[0] for # of elements
  s_mul_i32 s[6], s[0], 0x4
  s_mul_i32 s[10], s[0], 0x4
  s_mov_b32 s[7], 0x20000
  s_mov_b32 s[11], 0x20000
  v_lshl_add_u32 v0, s2, 8, v0       //s2 holds workgroup ID, g_tId = (workgroupId * workgroupSize) + tId
  v_cmpx_lt_u32 vcc, v0, s0          //set exec mask
  v_lshlrev_b32 v2, 2, v0            //byte offset: g_tId * sizeof(float)
  s_waitcnt lgkmcnt(0)               //wait for Srds
  buffer_load_dword v0, v2, s[4:7], 0 offen offset:0
  s_waitcnt vmcnt(0)                 //wait for buffer load
  v_max_f32 v0, v0, 0
  buffer_store_dword v0, v2, s[8:11], 0 offen offset:0
  s_endpgm
.Lrelu_end0:
  .size relu, .Lrelu_end0 - relu

.rodata
.p2align 6
.amdhsa_kernel relu
  .amdhsa_user_sgpr_kernarg_segment_ptr 1
  .amdhsa_system_sgpr_workgroup_id_x 1
  .amdhsa_accum_offset 4
  .amdhsa_next_free_vgpr .amdgcn.next_free_vgpr
  .amdhsa_next_free_sgpr .amdgcn.next_free_sgpr
.end_amdhsa_kernel

.text
.globl leaky_relu
.p2align 8
.type leaky_relu,@function
leaky_relu:
  s_load_dwordx2 s[4:5], s[0:1] 0    //s[4:7] for Srd a
  s_load_dwordx2 s[8:9], s[0:1] 8    //s[8:11] for Srd b
  s_load_dword s12, s[0:1] 16        //s12 to store alpha
  s_load_dword s0, s[0:1] 20         //reuse s0 for # of elements
  s_mul_i32 s6, s0, 0x4
  s_mul_i32 s10, s0, 0x4
  s_mov_b32 s7, 0x20000
  s_mov_b32 s11, 0x20000
  v_lshl_add_u32 v0, s2, 8, v0       //s2 holds workgroup ID, g_tId = (workgroupId * workgroupSize) + tId
  v_cmpx_lt_u32 vcc, v0, s0          //set exec mask
  v_lshlrev_b32 v2, 2, v0            //byte offset: g_tId * sizeof(float)
  s_waitcnt lgkmcnt(0)               //wait for Srds
  buffer_load_dword v0, v2, s[4:7], 0 offen offset:0
  s_waitcnt vmcnt(0)                 //wait for buffer load
  v_cmp_lt_f32 vcc, v0, 0
  s_and_saveexec_b64 s[14:15], vcc
  v_mul_f32 v0, v0, s12
  s_mov_b64 exec, s[14:15]
  buffer_store_dword v0, v2, s[8:11], 0 offen offset:0
  s_endpgm
.Lleaky_relu_end0:
  .size leaky_relu, .Lleaky_relu_end0 - leaky_relu

.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0

.rodata
.amdhsa_kernel leaky_relu
  .amdhsa_user_sgpr_kernarg_segment_ptr 1
  .amdhsa_system_sgpr_workgroup_id_x 1
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
 - .name: relu
   .symbol: relu.kd
   .kernarg_segment_size: 32
   .group_segment_fixed_size: 0
   .private_segment_fixed_size: 0
   .kernarg_segment_align: 8
   .wavefront_size: 64
   .sgpr_count: 12
   .vgpr_count: 2
   .agpr_count: 0
   .max_flat_workgroup_size: 256
   .args:
     - .size: 8
       .offset: 0
       .value_kind: global_buffer
       .address_space: global
     - .size: 8
       .offset: 8
       .value_kind: global_buffer
       .address_space: global
     - .size: 4
       .offset: 16
       .value_kind: by_value
 - .name: leaky_relu
   .symbol: leaky_relu.kd
   .kernarg_segment_size: 32
   .group_segment_fixed_size: 0
   .private_segment_fixed_size: 0
   .kernarg_segment_align: 8
   .wavefront_size: 64
   .sgpr_count: 16
   .vgpr_count: 2
   .agpr_count: 0
   .max_flat_workgroup_size: 256
   .args:
     - .size: 8
       .offset: 0
       .value_kind: global_buffer
       .address_space: global
     - .size: 8
       .offset: 8
       .value_kind: global_buffer
       .address_space: global
     - .size: 4
       .offset: 16
       .value_kind: by_value
     - .size: 4
       .offset: 20
       .value_kind: by_value
.end_amdgpu_metadata