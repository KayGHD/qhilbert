cmake_minimum_required(VERSION 3.21)
project(qhilbert_python LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(nanobind CONFIG REQUIRED)

nanobind_add_module(
    _hilbert
    STABLE_ABI
    NB_STATIC
    ext/hilbert_ext.cpp
)

install(TARGETS _hilbert LIBRARY DESTINATION qhilbert)
