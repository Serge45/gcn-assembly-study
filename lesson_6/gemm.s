.amdgcn_target "amdgcn-amd-amdhsa--gfx90a:xnack-"
.text
.globl gemm
.p2align 8
.type gemm,@function
gemm:
  //constants
  .set bpe, 4
  .set scalarBpe, 4
  .set numWorkgroups, 256
  .set tileM, 16
  .set tileN, 16
  .set depthU, 1
  //sgpr
  .set srdA, 4
  .set srdB, 8
  .set srdC, 4
  .set srdD, 8
  .set wgIdX, 2
  .set wgIdY, 3
  .set m, 12
  .set n, 13
  .set k, 14
  .set tmp, 21
  .set kIdx, 22
  .set rowIdx, 15
  .set colIdx, 27
  .set loadOffsetA, 16
  .set loadOffsetB, 24
  .set loadOffsetC, 16
  .set loadOffsetD, 24
  .set strideA0, 17
  .set strideA1, 18
  .set strideB0, 19
  .set strideB1, 20
  .set strideCD0, 24
  .set strideCD1, 25
  .set alpha, 23
  .set beta, 26
  //vgpr
  .set tId, 0
  .set glOffsetA, 1
  .set glOffsetB, 4
  .set glOffsetC, 1
  .set glOffsetD, 4
  .set tRow, 7
  .set tCol, 8
  .set aData, 2
  .set bData, 3
  .set cData, 3
  .set dData, 2
  .set accData, 5
  .set vTmp, 6
label_load_args:
  s_load_dwordx2 s[srdA:srdA+1], s[0:1] 0    //s[4:7] for Srd a
  s_load_dwordx2 s[srdB:srdB+1], s[0:1] 8    //s[8:11] for Srd b
  s_mov_b32 s[srdA+3], 0x20000
  s_mov_b32 s[srdB+3], 0x20000
  s_load_dword s[m], s[0:1] 32
  s_load_dword s[n], s[0:1] 36
  s_load_dword s[k], s[0:1] 40
  s_load_dword s[alpha], s[0:1] 44
  s_load_dword s[beta], s[0:1] 48
  s_waitcnt lgkmcnt(0)                       //wait for Srds
label_setup_input_srds:
  s_lshl_b32 s[rowIdx], s[wgIdX], 4          //row index for wg, wgIdX * tileM, e.g. for wg(2,2), row starts from 2 * 16
  s_mul_i32 s[tmp], s[m], s[k]               //compute # of elements for A
  s_lshl_b32 s[srdA+2], s[tmp], 2            //setup Srd A
  s_mul_i32 s[tmp], s[n], s[k]               //compute # of elements for B
  s_lshl_b32 s[srdB+2], s[tmp], 2            //setup Srd B
  s_lshl_b32 s[colIdx], s[wgIdY], 4          //col start idx
  s_mov_b32 s[strideA0], 1                   //col-maj
  s_mov_b32 s[strideB0], 1                   //col-maj
  s_mov_b32 s[strideA1], s[m]                //col-maj
  s_mov_b32 s[strideB1], s[k]                //col-maj
  s_mov_b32 s[strideCD0], 1                  //col-maj
  s_mov_b32 s[strideCD1], s[m]               //col-maj
  s_mul_i32 s[loadOffsetA], s[rowIdx], s[strideA0]
  s_lshl_b32 s[loadOffsetA], s[loadOffsetA], 2
  s_mul_i32 s[loadOffsetB], s[colIdx], s[strideB1]
  s_lshl_b32 s[loadOffsetB], s[loadOffsetB], 2
  v_mov_b32 v[accData], 0                    //init acc buf
  s_mov_b32 s[kIdx], 0                       //kIdx = 0
  v_and_b32 v[tRow], v[tId], tileM - 1       //local row for thread
  v_lshrrev_b32 v[tCol], 4, v[tId]           //local col for thread
  v_mov_b32 v[glOffsetA], s[kIdx]
  v_mul_lo_u32 v[glOffsetA], v[glOffsetA], s[strideA1]
  v_add_u32 v[glOffsetA], v[glOffsetA], v[tRow]
  v_lshlrev_b32 v[glOffsetA], 2, v[glOffsetA]     //load offset in byte for A
  v_mov_b32 v[glOffsetB], v[tCol]
  v_mul_lo_u32 v[glOffsetB], v[glOffsetB], s[strideB1]
  v_add_u32 v[glOffsetB], v[glOffsetB], s[kIdx]
  v_lshlrev_b32 v[glOffsetB], 2, v[glOffsetB]     //load offset in byte for B
label_outer_loop:
  buffer_load_dword v[aData], v[glOffsetA], s[srdA:srdA+3], s[loadOffsetA] offen offset:0
  s_add_u32 s[kIdx], s[kIdx], depthU
  v_lshl_add_u32 v[glOffsetA], s[m], 2, v[glOffsetA]
  buffer_load_dword v[bData], v[glOffsetB], s[srdB:srdB+3], s[loadOffsetB] offen offset:0
  v_add_u32 v[glOffsetB], v[glOffsetB], bpe
  s_waitcnt vmcnt(0)
  v_fma_f32 v[accData], v[aData], v[bData], v[accData]
  s_cmp_lt_u32 s[kIdx], s[k]
  s_cbranch_scc1 label_outer_loop
label_load_output_srds:
  s_load_dwordx2 s[srdC:srdC+1], s[0:1] 16    //s[4:7] for Srd c
  s_load_dwordx2 s[srdD:srdD+1], s[0:1] 24    //s[8:11] for Srd d
  s_mov_b32 s[srdC+3], 0x20000
  s_mov_b32 s[srdD+3], 0x20000
  s_mul_i32 s[tmp], s[m], s[n]
  s_lshl_b32 s[srdC+2], s[tmp], 2
  s_lshl_b32 s[srdD+2], s[tmp], 2
label_setup_output_offsets:
  s_mul_i32 s[loadOffsetC], s[colIdx], s[strideCD1]
  s_add_i32 s[loadOffsetC], s[loadOffsetC], s[rowIdx]
  s_lshl_b32 s[loadOffsetC], s[loadOffsetC], 2
  s_mov_b32 s[loadOffsetD], s[loadOffsetC]
  v_mul_lo_u32 v[glOffsetC], v[tCol], s[strideCD1]
  v_add_u32 v[glOffsetC], v[glOffsetC], v[tRow]
  v_lshlrev_b32 v[glOffsetC], 2, v[glOffsetC]
  v_mov_b32 v[glOffsetD], v[glOffsetC]
  s_waitcnt lgkmcnt(0)                       //wait for Srds, alpha and beta
label_load_c:
  buffer_load_dword v[cData], v[glOffsetC], s[srdC:srdC+3], s[loadOffsetC] offen offset:0
  v_mul_f32 v[accData], v[accData], s[alpha]
  s_waitcnt vmcnt(0)
  v_fma_f32 v[accData], s[beta], v[cData], v[accData]
label_write_d:
  buffer_store_dword v[accData], v[glOffsetD], s[srdD:srdD+3], s[loadOffsetD] offen offset:0
label_endpgm:
  s_endpgm
.Lgemm_end0:
  .size gemm, .Lgemm_end0 - gemm

.rodata
.p2align 6
.amdhsa_kernel gemm
  .amdhsa_user_sgpr_kernarg_segment_ptr 1
  .amdhsa_system_sgpr_workgroup_id_x 1
  .amdhsa_system_sgpr_workgroup_id_y 1
  .amdhsa_accum_offset 8
  .amdhsa_next_free_vgpr .amdgcn.next_free_vgpr
  .amdhsa_next_free_sgpr .amdgcn.next_free_sgpr
.end_amdhsa_kernel

.amdgpu_metadata
---
amdhsa.version:
 - 1
 - 1

amdhsa.kernels:
 - .name: gemm
   .symbol: gemm.kd
   .kernarg_segment_size: 56
   .group_segment_fixed_size: 0
   .private_segment_fixed_size: 0
   .kernarg_segment_align: 8
   .wavefront_size: 64
   .sgpr_count: 28
   .vgpr_count: 9
   .agpr_count: 0
   .max_flat_workgroup_size: 256
   .args:
     - .size: 8
       .offset: 0
       .value_kind: global_buffer
       .address_space: global
       .name: a_buf
     - .size: 8
       .offset: 8
       .value_kind: global_buffer
       .address_space: global
       .name: b_buf
     - .size: 8
       .offset: 16
       .value_kind: global_buffer
       .address_space: global
       .name: c_buf
     - .size: 8
       .offset: 24
       .value_kind: global_buffer
       .address_space: global
       .name: d_buf
     - .size: 4
       .offset: 32
       .value_kind: by_value
       .name: m
     - .size: 4
       .offset: 36
       .value_kind: by_value
       .name: n
     - .size: 4
       .offset: 40
       .value_kind: by_value
       .name: k
     - .size: 4
       .offset: 44
       .value_kind: by_value
       .name: alpha
     - .size: 4
       .offset: 48
       .value_kind: by_value
       .name: beta
.end_amdgpu_metadata