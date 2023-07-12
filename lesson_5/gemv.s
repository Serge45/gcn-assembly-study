.amdgcn_target "amdgcn-amd-amdhsa--gfx90a:xnack-"
.text
.globl gemv
.p2align 8
.type gemv,@function
gemv:
  //constants
  .set bpe, 4
  .set scalarBpe, 4
  .set numWorkgroups, 256
  .set tileM, numWorkgroups
  .set depthN, 32 
  .set vectorWidth, 4
  .set glVectorWidth, 4
  .set lds_a_offset, 0
  .set lds_x_offset, tileM * depthN * bpe
  //sgpr
  .set srdA, 4
  .set srdX, 8
  .set srdY, 4
  .set srdOut, 8
  .set wgId, 2
  .set m, 12
  .set n, 13
  .set numElements, 3
  .set rowIdx, 14
  .set colIdx, 15
  .set loadOffsetA, 16
  .set loadOffsetX, 16
  .set loadOffsetY, 16
  .set writeOffsetOut, 16
  .set strideM, 17
  .set strideN, 18
  .set alpha, 19
  .set beta, 20
  .set glCounter, 21
  .set lrCounter, 21
  .set tileMIdx, 22
  //vgpr
  .set tId, 0
  .set glOffset, 1
  .set aData, 2
  .set xData, aData + vectorWidth //6
  .set yData, xData + vectorWidth //10
  .set accData, yData + vectorWidth //14
  .set localColIdx, accData + 1
  .set localRowIdx, localColIdx + 1
  .set localWirteOffset, localRowIdx + 1
  .set lrOffset, 18
label_load_args:
    s_load_dwordx2 s[srdA:srdA+1], s[0:1] 0    //s[4:7] for Srd a
    s_load_dwordx2 s[srdX:srdX+1], s[0:1] 8    //s[8:11] for Srd x
    s_mov_b32 s[srdA+3], 0x20000
    s_mov_b32 s[srdX+3], 0x20000
    s_load_dword s[m], s[0:1] 24
    s_load_dword s[n], s[0:1] 28
    s_waitcnt lgkmcnt(0)                       //wait for Srds
label_setup_srds:
    s_mul_i32 s[rowIdx], s[wgId], numWorkgroups//row index for wg, wgId * numWorkgroups, e.g. for wg2, row starts from 2 * 256
    s_mul_i32 s[numElements], s[m], s[n]       //compute # of elements
    s_mul_i32 s[srdA+2], s[numElements], bpe   //setup Srd A
    s_mul_i32 s[srdX+2], s[n], bpe             //setup Srd X
    s_mov_b32 s[colIdx], 0                     //col start idx
    v_mov_b32 v[accData], 0                    //init acc buf
    s_mov_b32 s[strideM], bpe
    s_mul_i32 s[strideN], bpe, s[m]
    s_mov_b32 s[tileMIdx], tileM
label_outer_loop:
    // load x first
    v_mul_lo_u32 v[glOffset], v[tId], bpe
    s_mul_i32 s[loadOffsetX], s[colIdx], bpe
    buffer_load_dword v[xData], v[glOffset], s[srdX:srdX+3], s[loadOffsetX] offen offset:0
    s_mov_b32 s[glCounter], 0
    s_waitcnt vmcnt(0)
    ds_write_b32 v[glOffset], v[xData] offset:lds_x_offset
label_global_read:
    s_mul_i32 s[loadOffsetA], s[colIdx], s[m]
    s_add_u32 s[loadOffsetA], s[loadOffsetA], s[rowIdx]
    s_mul_i32 s[loadOffsetA], s[loadOffsetA], bpe
    v_lshrrev_b32 v[localColIdx], 6, v[tId]
    v_and_b32 v[localRowIdx], tileM / vectorWidth - 1, v[tId]
    v_lshlrev_b32 v[localRowIdx], 2, v[localRowIdx]
    v_mul_u32_u24 v[glOffset], v[localColIdx], s[m]
    v_add_u32 v[glOffset], v[glOffset], v[localRowIdx]
    v_lshlrev_b32 v[glOffset], 2, v[glOffset]
    buffer_load_dwordx4 v[aData:aData+3], v[glOffset], s[srdA:srdA+3], s[loadOffsetA] offen offset:0
    v_add_u32 v[lrOffset], s[glCounter], v[localColIdx]
    v_mul_lo_u32 v[lrOffset], s[tileMIdx], v[lrOffset]
    v_add_u32 v[lrOffset], v[lrOffset], v[localRowIdx]
    v_lshlrev_b32 v[lrOffset], 2, v[lrOffset]
    s_waitcnt vmcnt(0)
    ds_write_b128 v[lrOffset], v[aData:aData+3] offset:lds_a_offset
    s_add_u32 s[colIdx], s[colIdx], glVectorWidth
    s_add_u32 s[glCounter], s[glCounter], glVectorWidth
    s_cmp_lt_u32 s[glCounter], depthN
    s_cbranch_scc1 label_global_read
label_global_read_end:
    s_waitcnt lgkmcnt(0)
    s_barrier
    v_mov_b32 v[localRowIdx], v[tId]
    s_mov_b32 s[lrCounter], 0
label_local_read:
    v_mov_b32 v[glOffset], s[lrCounter]
    v_mul_u32_u24 v[glOffset], tileM, v[glOffset]
    v_add_u32 v[glOffset], v[glOffset], v[tId]
    v_mul_lo_u32 v[glOffset], v[glOffset], bpe
    ds_read_b32 v[aData], v[glOffset] offset:0
    ds_read_b32 v[aData+1], v[glOffset] offset:1 * tileM * bpe
    ds_read_b32 v[aData+2], v[glOffset] offset:2 * tileM * bpe
    ds_read_b32 v[aData+3], v[glOffset] offset:3 * tileM * bpe
    v_mul_lo_u32 v[glOffset], s[lrCounter], bpe
    ds_read_b128 v[xData:xData+vectorWidth-1], v[glOffset] offset:lds_x_offset
    s_waitcnt lgkmcnt(0)
label_local_read_end:
label_matmul_begin:
    v_fma_f32 v[accData], v[aData], v[xData], v[accData]
    v_fma_f32 v[accData], v[aData+1], v[xData+1], v[accData]
    v_fma_f32 v[accData], v[aData+2], v[xData+2], v[accData]
    v_fma_f32 v[accData], v[aData+3], v[xData+3], v[accData]
label_matmul_end:
    s_add_u32 s[lrCounter], s[lrCounter], vectorWidth
    s_cmp_lt_u32 s[lrCounter], depthN
    s_cbranch_scc1 label_local_read
    s_cmp_lt_u32 s[colIdx], s[n]
    s_cbranch_scc1 label_outer_loop
label_outer_loop_end:
    s_load_dwordx2 s[srdY:srdY+1], s[0:1] 16       //s[4:7] for Srd y
    s_load_dwordx2 s[srdOut:srdOut+1], s[0:1] 40   //s[8:11] for Srd out
    s_mov_b32 s[srdY+3], 0x20000
    s_mov_b32 s[srdOut+3], 0x20000
    s_mul_i32 s[srdY+2], s[m], bpe
    s_mul_i32 s[srdOut+2], s[m], bpe
    s_mul_i32 s[loadOffsetY], s[rowIdx], bpe
    v_mul_lo_u32 v[glOffset], v[tId], bpe
    s_waitcnt lgkmcnt(0)
    buffer_load_dword v[yData], v[glOffset], s[srdY:srdY+3], s[loadOffsetY] offen offset:0
    s_load_dword s[alpha], s[0:1] 32
    s_load_dword s[beta], s[0:1] 36
    s_waitcnt vmcnt(0), lgkmcnt(0)
    v_mul_f32 v[yData], v[yData], s[beta]
    v_fma_f32 v[accData], s[alpha], v[accData], v[yData]
    s_mul_i32 s[writeOffsetOut], s[rowIdx], bpe
    buffer_store_dword v[accData], v[glOffset], s[srdOut:srdOut+3], s[writeOffsetOut] offen offset:0
  s_endpgm
.Lgemv_end0:
  .size gemv, .Lgemv_end0 - gemv

.rodata
.p2align 6
.amdhsa_kernel gemv
  .amdhsa_user_sgpr_kernarg_segment_ptr 1
  .amdhsa_system_sgpr_workgroup_id_x 1
  .amdhsa_accum_offset 16
  .amdhsa_next_free_vgpr .amdgcn.next_free_vgpr
  .amdhsa_next_free_sgpr .amdgcn.next_free_sgpr
.end_amdhsa_kernel

.amdgpu_metadata
---
amdhsa.version:
 - 1
 - 1

amdhsa.kernels:
 - .name: gemv
   .symbol: gemv.kd
   .kernarg_segment_size: 48
   .group_segment_fixed_size: 0
   .private_segment_fixed_size: 0
   .kernarg_segment_align: 8
   .wavefront_size: 64
   .sgpr_count: 20
   .vgpr_count: 16
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
       .name: x_buf
     - .size: 8
       .offset: 16
       .value_kind: global_buffer
       .address_space: global
       .name: y_buf
     - .size: 4
       .offset: 24
       .value_kind: by_value
       .name: m
     - .size: 4
       .offset: 28
       .value_kind: by_value
       .name: n
     - .size: 4
       .offset: 32
       .value_kind: by_value
       .name: alpha
     - .size: 4
       .offset: 36
       .value_kind: by_value
       .name: beta
     - .size: 8
       .offset: 40
       .value_kind: global_buffer
       .address_space: global
       .name: out_buf
.end_amdgpu_metadata