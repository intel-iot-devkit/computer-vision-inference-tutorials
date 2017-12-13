# Copyright (c) 2017 Intel Corporation

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#      http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

cmake_minimum_required (VERSION 2.8)

project(Samples)

####################################
## to use C++11
set (CMAKE_CXX_STANDARD 11)
set (CMAKE_CXX_STANDARD_REQUIRED ON)
####################################

if("${CMAKE_SIZEOF_VOID_P}" EQUAL "8")
    set (ARCH intel64)
else()
    set (ARCH ia32)
endif()

if("${CMAKE_BUILD_TYPE}" STREQUAL "")
    message(STATUS "CMAKE_BUILD_TYPE not defined, 'Release' will be used")
    set(CMAKE_BUILD_TYPE "Release")
endif()

#linux OS name detection
if (UNIX AND NOT(LIB_FOLDER))
    SET(LIB_DL dl)

    if (NOT EXISTS "/etc/lsb-release")
        execute_process(COMMAND find /etc/ -maxdepth 1 -type f -name *-release -exec cat {} \;
                    OUTPUT_VARIABLE release_data RESULT_VARIABLE result)
        set(name_regex "NAME=\"([^ \"\n]*).*\"\n")
        set(version_regex "VERSION=\"([0-9]+(\\.[0-9]+)?)[^\n]*\"")
    else()
        #linux version detection using cat /etc/lsb-release
        file(READ "/etc/lsb-release" release_data)
        set(name_regex "DISTRIB_ID=([^ \n]*)\n")
        set(version_regex "DISTRIB_RELEASE=([0-9]+(\\.[0-9]+)?)")
    endif()

    string(REGEX MATCH ${name_regex} name ${release_data})
    set(os_name ${CMAKE_MATCH_1})

    string(REGEX MATCH ${version_regex} version ${release_data})
    set(os_name "${os_name} ${CMAKE_MATCH_1}")

    if (NOT os_name)
        message(FATAL_ERROR "Cannot detect OS via reading /etc/*-release:\n ${release_data}")
    endif()

    message (STATUS "/etc/*-release distrib: ${os_name}")

    if (${os_name} STREQUAL "Ubuntu 14.04")
        set(OS_LIB_FOLDER "ubuntu_14.04/")
    elseif (${os_name} STREQUAL "Ubuntu 16.04")
        set(OS_LIB_FOLDER "ubuntu_16.04/")
    elseif (${os_name} STREQUAL "CentOS 7")
        set(OS_LIB_FOLDER "centos_7.2/")
    else()
        message(FATAL_ERROR "${os_name} is not supported. List of supported OS: Ubuntu 14.04, Ubuntu 16.04, CentOS 7")
    endif()

    add_definitions(-DOS_LIB_FOLDER="${OS_LIB_FOLDER}")
endif()

if (NOT(BIN_FOLDER))
    set (BIN_FOLDER ../bin/${ARCH})
endif()

if (NOT(LIB_FOLDER))
    set (LIB_FOLDER ../lib/${OS_LIB_FOLDER}${ARCH})
endif()

if (NOT ENABLE_SAMPLES_CORE AND NOT ENABLE_SAMPLES)
    set(ENABLE_SAMPLES_CORE ON)
    set(ENABLE_SAMPLES ON)
endif()

set(CMAKE_CXX_FLAGS "-std=c++11 ${CMAKE_CXX_FLAGS}")

if(NOT(UNIX))
    set (CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/${BIN_FOLDER})
    set (CMAKE_LIBRARY_PATH ${CMAKE_SOURCE_DIR}/${BIN_FOLDER})
    set (CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/${BIN_FOLDER})
    set (CMAKE_COMPILE_PDB_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/${BIN_FOLDER})
    set (CMAKE_PDB_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/${BIN_FOLDER})
    set (CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/${BIN_FOLDER})
    set (LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/${BIN_FOLDER})
    set (LIBRARY_OUTPUT_PATH ${LIBRARY_OUTPUT_DIRECTORY}) # compatibility issue: linux uses LIBRARY_OUTPUT_PATH, windows uses LIBRARY_OUTPUT_DIRECTORY
else ()
    set (CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/${BIN_FOLDER}/${CMAKE_BUILD_TYPE}/lib)
    set (CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/${BIN_FOLDER}/${CMAKE_BUILD_TYPE}/lib)
    set (CMAKE_COMPILE_PDB_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/${BIN_FOLDER}/${CMAKE_BUILD_TYPE})
    set (CMAKE_PDB_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/${BIN_FOLDER}/${CMAKE_BUILD_TYPE})
    set (CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/${BIN_FOLDER}/${CMAKE_BUILD_TYPE})
    set (LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/${BIN_FOLDER}/${CMAKE_BUILD_TYPE}/lib)
    set (LIBRARY_OUTPUT_PATH ${LIBRARY_OUTPUT_DIRECTORY}/lib)
endif()

if (WIN32)
    if(NOT "${CMAKE_GENERATOR}" MATCHES "(Win64|IA64)")
        message(FATAL_ERROR "Only 64-bit supported on Windows")
    endif()

    set_property(DIRECTORY APPEND PROPERTY COMPILE_DEFINITIONS _CRT_SECURE_NO_WARNINGS)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_SCL_SECURE_NO_WARNINGS")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /EHsc") #no asynchronous structured exception handling
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /LARGEADDRESSAWARE")
    if (ENABLE_OMP)
        find_package(OpenMP)
        if (OPENMP_FOUND)
            set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
            set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
        endif()
    endif()
else()
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror -Werror=return-type ")
	if (APPLE)
		set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-error=unused-command-line-argument")
	elseif(UNIX)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wuninitialized -Winit-self -Wmaybe-uninitialized")
	endif()
endif()

include("${CMAKE_SOURCE_DIR}/cmake/feature_defs.cmake" OPTIONAL)

add_subdirectory(thirdparty/gflags)

add_subdirectory(format_reader)
#add_subdirectory(object_detection_sample)
add_subdirectory(object_detection_sample_ssd)
add_subdirectory(classification_sample)
add_subdirectory(extensibility_sample)
#add_subdirectory(segmentation_sample)
#add_subdirectory(multi_output_sample)

message(STATUS "Validation app build is switched off")
