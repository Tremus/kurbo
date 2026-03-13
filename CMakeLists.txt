cmake_minimum_required(VERSION 3.15)
project(curbo LANGUAGES C)
set(CMAKE_C_VERISON 99)

add_compile_options(
    -Werror=absolute-value
    -Werror=return-type
    -Werror=shadow
    -Werror=incompatible-pointer-types
    -Werror=parentheses
    -Werror=excess-initializers
    -Werror=array-bounds
    # -Werror=implicit-fallthrough
    -Wno-unused-function
    -Wno-unused-variable

    -Wno-deprecated-declarations
    -Wno-deprecated
    -Wno-multichar
    -Wno-nullability-completeness
    -Wno-writable-strings
    -Wno-c2x-extensions
    -Wno-c++14-extensions
    -Wno-c++17-extensions
    -Wno-c++20-extensions
    -Wno-microsoft-enum-forward-reference
    -Wno-typedef-redefinition
    -Wno-visibility
)

add_executable(curbo curbo.c)

