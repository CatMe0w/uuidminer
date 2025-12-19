cmake_minimum_required(VERSION 3.18)

project(uuidminer LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(USE_CUDA "Enable CUDA support" ON)

# Source files
set(SOURCES
    uuidminer/common.h
)

if(USE_CUDA)
    find_package(CUDAToolkit REQUIRED)
    list(APPEND SOURCES
        uuidminer/kernel.cu
    )
    add_compile_definitions(USE_CUDA)
endif()

add_executable(uuidminer ${SOURCES})

target_include_directories(uuidminer PRIVATE uuidminer)

if(USE_CUDA)
    # Link against CUDA toolkit to ensure IntelliSense finds headers
    target_link_libraries(uuidminer PRIVATE CUDA::cudart_static)

    # Enable Relocatable Device Code (matches VS GenerateRelocatableDeviceCode=true)
    set_target_properties(uuidminer PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    
    # Enable Fast Math in Release (matches VS FastMath=true)
    target_compile_options(uuidminer PRIVATE $<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CUDA>>:-use_fast_math>)
    
    # Generate line info for Nsight profiling
    target_compile_options(uuidminer PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)
endif()

# Windows/Visual Studio specific settings
if(MSVC)
    # Use static runtime (MT/MTd) to match original project and avoid LNK4098 with cudart_static
    set_property(TARGET uuidminer PROPERTY MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")

    # Ensure PDB generation in Release for Nsight debugging/profiling
    target_compile_options(uuidminer PRIVATE $<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CXX>>:/Zi>)
    target_compile_options(uuidminer PRIVATE $<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CUDA>>:-Xcompiler=/Zi>)
    
    # Use CMAKE_EXE_LINKER_FLAGS to avoid flag splitting issues with target_link_options in CUDA builds
    set(CMAKE_EXE_LINKER_FLAGS_RELEASE "${CMAKE_EXE_LINKER_FLAGS_RELEASE} /DEBUG /OPT:REF /OPT:ICF")
    
    # Set startup project
    set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT uuidminer)
endif()
