.amdgcn_target "amdgcn-amd-amdhsa--gfx90a:xnack-"
.text
.globl gemm_mfma
.p2align 8
.type gemm_mfma,@function
gemm_mfma:
  //constants
  .set bpe, 4
  .set scalarBpe, 4
  .set wavefrontSize, 64
  .set numWorkitems, 256
  .set waveM, 2
  .set waveN, 2
  .set miM, 16
  .set miN, 16
  .set miK, 4
  .set tileM, miM * waveM   //32
  .set tileN, miN * waveN   //32
  .set depthU, 16
  .set glvw, 2//32*16/256=2, same shape of A B tile
  .set ldsStrideA, tileM
  .set ldsStrideB, depthU
  .set ldsOffsetB, tileM * depthU * bpe
  .set prefetchLdsOffset, ldsOffsetB + tileN * depthU * bpe
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
  .set kernelArg, 28 //s[28:41]
  .set ldsStartAddr, 42
  //vgpr
  .set tId, 0
  .set glOffsetA, 1
  .set glOffsetB, 4
  .set glOffsetC, 1
  .set glOffsetD, 4
  .set vTmp, 6
  .set tRow, 7
  .set tCol, 8
  .set g2lA, 10           //v[10:11]
  .set g2lB, 12           //v[12:13]
  .set ldsWriteAddrA, 14
  .set ldsWriteAddrB, 15
  .set ldsReadAddrA, 16
  .set ldsReadAddrB, 17
  .set valuA, 18
  .set valuB, 19
  .set waveId, 20
  .set wtId, 21
  .set wRow, 22
  .set wCol, 23
  .set valuAcc, 24        //v[24:27]
  .set valuC, 28          //v[28:31]
  .set valuD, 32          //v[32:35]
  .set valuA1, 36
  .set valuB1, 37
  //acc
  .set accRes, 0          //a[0:3]
label_load_args:
  s_load_dwordx4 s[kernelArg:kernelArg+3], s[0:1], 0
  s_load_dwordx4 s[kernelArg+4:kernelArg+7], s[0:1], 16
  s_load_dwordx4 s[kernelArg+8:kernelArg+11], s[0:1], 32
  s_load_dwordx2 s[kernelArg+12:kernelArg+13], s[0:1], 48
  s_mov_b32 s[srdA+3], 0x20000
  s_mov_b32 s[srdB+3], 0x20000
  s_waitcnt lgkmcnt(0)                       //wait for all args
  s_mov_b64 s[srdA:srdA+1], s[kernelArg:kernelArg+1]
  s_mov_b64 s[srdB:srdB+1], s[kernelArg+2:kernelArg+3]
  s_mov_b32 s[alpha], s[kernelArg+11]
  s_mov_b32 s[beta], s[kernelArg+12]
  s_mov_b64 s[m:m+1], s[kernelArg+8:kernelArg+9]
  s_mov_b32 s[k], s[kernelArg+10]
label_setup_input_srds:
  s_lshl_b32 s[rowIdx], s[wgIdX], 5          //row index for wg, wgIdX * tileM, e.g. for wg(2,2), row starts from 2 * 32
  s_mul_i32 s[tmp], s[m], s[k]               //compute # of elements for A
  s_lshl_b32 s[srdA+2], s[tmp], 2            //setup Srd A
  s_mul_i32 s[tmp], s[n], s[k]               //compute # of elements for B
  s_lshl_b32 s[srdB+2], s[tmp], 2            //setup Srd B
  s_lshl_b32 s[colIdx], s[wgIdY], 5          //col start idx
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
  s_mov_b32 s[kIdx], 0                       //kIdx = 0
  v_accvgpr_write_b32 a0, 0
  v_accvgpr_write_b32 a1, 0
  v_accvgpr_write_b32 a2, 0
  v_accvgpr_write_b32 a3, 0
label_vm_read_prefetch:
  //addr for vm read A
  v_and_b32 v[tRow], v[tId], 15          //tRow = (tId % 16) * vw(2)
  v_lshlrev_b32 v[tRow], 1, v[tRow]      //tRow = (tId % 16) * vw(2)
  v_lshrrev_b32 v[tCol], 4, v[tId]       //tCol = tId / 16
  v_mul_lo_u32 v[glOffsetA], v[tCol], s[strideA1]
  v_add_u32 v[glOffsetA], v[glOffsetA], v[tRow]
  v_lshlrev_b32 v[glOffsetA], 2, v[glOffsetA]
  //addr for vm read B
  v_and_b32 v[tRow], v[tId], 7
  v_lshlrev_b32 v[tRow], 1, v[tRow]
  v_lshrrev_b32 v[tCol], 3, v[tId]
  v_mul_lo_u32 v[glOffsetB], v[tCol], s[strideB1]
  v_add_u32 v[glOffsetB], v[glOffsetB], v[tRow]
  v_lshlrev_b32 v[glOffsetB], 2, v[glOffsetB]
  buffer_load_dwordx2 v[g2lA:g2lA+1], v[glOffsetA], s[srdA:srdA+3], s[loadOffsetA] offen offset:0
  buffer_load_dwordx2 v[g2lB:g2lB+1], v[glOffsetB], s[srdB:srdB+3], s[loadOffsetB] offen offset:0
  //addr for lds write A
  v_and_b32 v[tRow], v[tId], 15
  v_lshlrev_b32 v[tRow], 1, v[tRow]
  v_lshrrev_b32 v[tCol], 4, v[tId]
  v_mul_lo_u32 v[ldsWriteAddrA], v[tCol], tileM
  v_add_u32 v[ldsWriteAddrA], v[ldsWriteAddrA], v[tRow]
  v_lshlrev_b32 v[ldsWriteAddrA], 2, v[ldsWriteAddrA]
  //addr for lds write B
  v_and_b32 v[tRow], v[tId], 7
  v_lshlrev_b32 v[tRow], 1, v[tRow]
  v_lshrrev_b32 v[tCol], 3, v[tId]
  v_mul_lo_u32 v[ldsWriteAddrB], v[tCol], depthU
  v_add_u32 v[ldsWriteAddrB], v[ldsWriteAddrB], v[tRow]
  v_lshlrev_b32 v[ldsWriteAddrB], 2, v[ldsWriteAddrB]
  s_waitcnt vmcnt(0)
  ds_write_b64 v[ldsWriteAddrA], v[g2lA:g2lA+1], offset:0
  ds_write_b64 v[ldsWriteAddrB], v[g2lB:g2lB+1], offset:ldsOffsetB
label_vm_read_addr_increment_prefetch:
  s_mul_i32 s[tmp], s[strideA1], depthU * bpe
  s_add_u32 s[loadOffsetA], s[loadOffsetA], s[tmp]
  s_add_u32 s[loadOffsetB], s[loadOffsetB], depthU * bpe
  s_mov_b32 s[ldsStartAddr], prefetchLdsOffset
  v_add_u32 v[ldsWriteAddrA], v[ldsWriteAddrA], s[ldsStartAddr]
  v_add_u32 v[ldsWriteAddrB], v[ldsWriteAddrB], s[ldsStartAddr]
label_lds_read_addr:
  v_lshrrev_b32 v[waveId], 6, v[tId]
  v_and_b32 v[wtId], wavefrontSize-1, v[tId]
  v_lshrrev_b32 v[wCol], 1, v[waveId]
  v_lshlrev_b32 v[wCol], 4, v[wCol]//MI_N * wId
  v_and_b32 v[wRow], 1, v[waveId]
  v_lshlrev_b32 v[wRow], 4, v[wRow]//MI_M * wId

  v_and_b32 v[tRow], 15, v[wtId]
  v_lshrrev_b32 v[tCol], 4, v[wtId]
  v_add_u32 v[tRow], v[tRow], v[wRow]
  v_add_u32 v[tCol], v[tCol], 0
  v_mul_lo_u32 v[ldsReadAddrA], v[tCol], ldsStrideA 
  v_add_u32 v[ldsReadAddrA], v[ldsReadAddrA], v[tRow]
  v_lshlrev_b32 v[ldsReadAddrA], 2, v[ldsReadAddrA]
label_lds_read_addr_b:
  v_and_b32 v[tCol], 15, v[wtId]
  v_lshrrev_b32 v[tRow], 4, v[wtId]
  v_add_u32 v[tRow], v[tRow], 0
  v_add_u32 v[tCol], v[tCol], v[wCol]
  v_mul_lo_u32 v[ldsReadAddrB], v[tCol], ldsStrideB 
  v_add_u32 v[ldsReadAddrB], v[ldsReadAddrB], v[tRow]
  v_lshlrev_b32 v[ldsReadAddrB], 2, v[ldsReadAddrB]
lable_sync_prefetch:
  s_waitcnt lgkmcnt(0)
  s_barrier
label_outer_loop:
label_vm_read:
  buffer_load_dwordx2 v[g2lA:g2lA+1], v[glOffsetA], s[srdA:srdA+3], s[loadOffsetA] offen offset:0
  buffer_load_dwordx2 v[g2lB:g2lB+1], v[glOffsetB], s[srdB:srdB+3], s[loadOffsetB] offen offset:0
label_lds_prefetch:
  ds_read_b32 v[valuA1], v[ldsReadAddrA], offset: 0
  ds_read_b32 v[valuB1], v[ldsReadAddrB], offset: ldsOffsetB 
label_unrolled_mac:
  ds_read_b32 v[valuA], v[ldsReadAddrA], offset: miK * ldsStrideA * bpe
  ds_read_b32 v[valuB], v[ldsReadAddrB], offset: ldsOffsetB + miK * bpe
  s_waitcnt lgkmcnt(2)

  //iter0
  v_mfma_f32_16x16x4f32 a[0:3], v[valuA1], v[valuB1], a[0:3]
  ds_read_b32 v[valuA1], v[ldsReadAddrA], offset: 2 * miK * ldsStrideA * bpe
  ds_read_b32 v[valuB1], v[ldsReadAddrB], offset: ldsOffsetB + 2 * miK * bpe
  s_waitcnt lgkmcnt(2)

  //iter1
  v_mfma_f32_16x16x4f32 a[0:3], v[valuA], v[valuB], a[0:3]
  ds_read_b32 v[valuA], v[ldsReadAddrA], offset: 3 * miK * ldsStrideA * bpe
  ds_read_b32 v[valuB], v[ldsReadAddrB], offset: ldsOffsetB + 3 * miK * bpe
  s_waitcnt lgkmcnt(2)

  //iter2
  v_mfma_f32_16x16x4f32 a[0:3], v[valuA1], v[valuB1], a[0:3]
  s_waitcnt lgkmcnt(0)
  s_waitcnt vmcnt(1)
  ds_write_b64 v[ldsWriteAddrA], v[g2lA:g2lA+1], offset:0

  //iter3
  v_mfma_f32_16x16x4f32 a[0:3], v[valuA], v[valuB], a[0:3]

label_lds_write:
  s_waitcnt vmcnt(0)
  ds_write_b64 v[ldsWriteAddrB], v[g2lB:g2lB+1], offset:ldsOffsetB
  v_add_i32 v[ldsReadAddrA], v[ldsReadAddrA], s[ldsStartAddr]
  v_add_i32 v[ldsReadAddrB], v[ldsReadAddrB], s[ldsStartAddr]
  s_mul_i32 s[ldsStartAddr], s[ldsStartAddr], -1
  v_add_i32 v[ldsWriteAddrA], v[ldsWriteAddrA], s[ldsStartAddr]
  v_add_i32 v[ldsWriteAddrB], v[ldsWriteAddrB], s[ldsStartAddr]

  s_add_u32 s[kIdx], s[kIdx], depthU
  s_mul_i32 s[tmp], s[strideA1], depthU * bpe;
  s_add_u32 s[loadOffsetA], s[loadOffsetA], s[tmp]
  s_add_u32 s[loadOffsetB], s[loadOffsetB], depthU * bpe
  s_add_u32 s[tmp], s[kIdx], depthU
  s_waitcnt lgkmcnt(0)
  s_barrier
  s_cmp_lt_u32 s[tmp], s[k]
  s_cbranch_scc1 label_outer_loop
label_prefetch_last_loop:
  ds_read_b32 v[valuA1], v[ldsReadAddrA], offset: 0
  ds_read_b32 v[valuB1], v[ldsReadAddrB], offset: ldsOffsetB 
  ds_read_b32 v[valuA], v[ldsReadAddrA], offset: miK * ldsStrideA * bpe
  ds_read_b32 v[valuB], v[ldsReadAddrB], offset: ldsOffsetB + miK * bpe
  s_waitcnt lgkmcnt(2)
  //iter0
  v_mfma_f32_16x16x4f32 a[0:3], v[valuA1], v[valuB1], a[0:3]
  ds_read_b32 v[valuA1], v[ldsReadAddrA], offset: 2 * miK * ldsStrideA * bpe
  ds_read_b32 v[valuB1], v[ldsReadAddrB], offset: ldsOffsetB + 2 * miK * bpe
  s_waitcnt lgkmcnt(2)
  //iter1
  v_mfma_f32_16x16x4f32 a[0:3], v[valuA], v[valuB], a[0:3]
  ds_read_b32 v[valuA], v[ldsReadAddrA], offset: 3 * miK * ldsStrideA * bpe
  ds_read_b32 v[valuB], v[ldsReadAddrB], offset: ldsOffsetB + 3 * miK * bpe

  s_waitcnt lgkmcnt(2)
  //iter2
  v_mfma_f32_16x16x4f32 a[0:3], v[valuA1], v[valuB1], a[0:3]

  //iter3
  s_waitcnt lgkmcnt(0)
  v_mfma_f32_16x16x4f32 a[0:3], v[valuA], v[valuB], a[0:3]

label_load_output_srds:
  s_mov_b64 s[srdC:srdC+1], s[kernelArg+4:kernelArg+5]
  s_mov_b64 s[srdD:srdD+1], s[kernelArg+6:kernelArg+7]
  s_mov_b32 s[srdC+3], 0x20000
  s_mov_b32 s[srdD+3], 0x20000
  s_mul_i32 s[tmp], s[m], s[n]
  s_lshl_b32 s[srdC+2], s[tmp], 2
  s_lshl_b32 s[srdD+2], s[tmp], 2
label_setup_output_base_offsets:
  s_mul_i32 s[loadOffsetC], s[colIdx], s[strideCD1]
  s_add_i32 s[loadOffsetC], s[loadOffsetC], s[rowIdx]
  s_lshl_b32 s[loadOffsetC], s[loadOffsetC], 2
  s_mov_b32 s[loadOffsetD], s[loadOffsetC]
label_setup_output_offsets:
  v_and_b32 v[tCol], 15, v[wtId]      //col = wtId % 16
  v_lshrrev_b32 v[tRow], 4, v[wtId]   //row = (wtId / 16) * 4
  v_lshlrev_b32 v[tRow], 2, v[tRow]   //row = (wtId / 16) * 4
  v_add_u32 v[tRow], v[tRow], v[wRow]
  v_add_u32 v[tCol], v[tCol], v[wCol]
  v_mul_lo_u32 v[glOffsetC], v[tCol], s[strideCD1]
  v_add_u32 v[glOffsetC], v[glOffsetC], v[tRow]
  v_lshlrev_b32 v[glOffsetC], 2, v[glOffsetC]
  v_mov_b32 v[glOffsetD], v[glOffsetC]
label_load_c:
  v_accvgpr_read_b32 v[valuAcc], a0
  v_accvgpr_read_b32 v[valuAcc+1], a1
  v_accvgpr_read_b32 v[valuAcc+2], a2
  v_accvgpr_read_b32 v[valuAcc+3], a3
  buffer_load_dwordx4 v[valuC:valuC+3], v[glOffsetC], s[srdC:srdC+3], s[loadOffsetC] offen offset:0
  v_mul_f32 v[valuAcc], v[valuAcc], s[alpha]
  v_mul_f32 v[valuAcc+1], v[valuAcc+1], s[alpha]
  v_mul_f32 v[valuAcc+2], v[valuAcc+2], s[alpha]
  v_mul_f32 v[valuAcc+3], v[valuAcc+3], s[alpha]
  s_waitcnt vmcnt(0)
  v_fma_f32 v[valuAcc], s[beta], v[valuC], v[valuAcc]
  v_fma_f32 v[valuAcc+1], s[beta], v[valuC+1], v[valuAcc+1]
  v_fma_f32 v[valuAcc+2], s[beta], v[valuC+2], v[valuAcc+2]
  v_fma_f32 v[valuAcc+3], s[beta], v[valuC+3], v[valuAcc+3]
label_write_d:
  buffer_store_dwordx4 v[valuAcc:valuAcc+3], v[glOffsetD], s[srdD:srdD+3], s[loadOffsetD] offen offset:0
label_endpgm:
  s_endpgm
.Lgemm_mfma_end0:
  .size gemm_mfma, .Lgemm_mfma_end0 - gemm_mfma

.rodata
.p2align 6
.amdhsa_kernel gemm_mfma
  .amdhsa_user_sgpr_kernarg_segment_ptr 1
  .amdhsa_system_sgpr_workgroup_id_x 1
  .amdhsa_system_sgpr_workgroup_id_y 1
  .amdhsa_accum_offset 40
  .amdhsa_group_segment_fixed_size 8192
  .amdhsa_next_free_vgpr 56//.amdgcn.next_free_vgpr
  .amdhsa_next_free_sgpr 43//.amdgcn.next_free_sgpr
.end_amdhsa_kernel

.amdgpu_metadata
---
amdhsa.version:
 - 1
 - 1

amdhsa.kernels:
 - .name: gemm_mfma
   .symbol: gemm_mfma.kd
   .kernarg_segment_size: 56
   .group_segment_fixed_size: 8192
   .private_segment_fixed_size: 0
   .kernarg_segment_align: 8
   .wavefront_size: 64
   .sgpr_count: 43
   .vgpr_count: 56
   .agpr_count: 4
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