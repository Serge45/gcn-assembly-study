function(build_asm f custom_target)
  add_custom_command(
    TARGET ${custom_target}
    PRE_BUILD
    COMMAND ${CMAKE_CXX_COMPILER} -x assembler -target amdgcn-amd-amdhsa -mcode-object-version=4 -mcpu=gfx942 -mwavefrontsize64 -c -g -o ${CMAKE_CURRENT_BINARY_DIR}/${f}.o ${CMAKE_CURRENT_SOURCE_DIR}/${f}.s
    COMMAND ${CMAKE_CXX_COMPILER} -target amdcgn-amdhsa ${CMAKE_CURRENT_BINARY_DIR}/${f}.o -o ${CMAKE_CURRENT_BINARY_DIR}/${f}.co
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/{f}.s
    COMMENT "Compiling ${CMAKE_CURRENT_SOURCE_DIR}/${f}.s to code object ${CMAKE_CURRENT_BINARY_DIR}/${f}.co"
    VERBATIM
  )
endfunction()

function(gen_and_build_asm f custom_target)
  add_custom_command(
    TARGET ${custom_target}
    PRE_BUILD
    COMMAND python ${CMAKE_CURRENT_SOURCE_DIR}/codegen.py -o ${CMAKE_CURRENT_BINARY_DIR}/${f}.s
    VERBATIM
  )
endfunction()
