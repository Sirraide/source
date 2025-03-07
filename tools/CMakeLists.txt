add_executable(vm-tablegen VMTablegen.cc)
target_link_libraries(vm-tablegen PRIVATE LLVMTableGen LLVMSupport)
