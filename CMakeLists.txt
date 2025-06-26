cmake_minimum_required(VERSION 3.12)
project(wfc-lv2 VERSION 1.0.0 DESCRIPTION "Wave Function Collapse LV2 Plugin")

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required packages
find_package(PkgConfig REQUIRED)
pkg_check_modules(LV2 REQUIRED lv2)

# Compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -fPIC -fvisibility=hidden")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -DNDEBUG")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -O0")

# Build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Source files
set(SOURCES
    src/wfc.cpp
)

# Check if source files exist
foreach(source ${SOURCES})
    if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${source}")
        message(WARNING "Source file ${source} not found. Please ensure all source files are present.")
    endif()
endforeach()

# Create shared library
add_library(wfc SHARED ${SOURCES})

# Set target properties
set_target_properties(wfc PROPERTIES
    PREFIX ""
    SUFFIX ".so"
    OUTPUT_NAME "wfc"
)

# Include directories
target_include_directories(wfc PRIVATE
    ${LV2_INCLUDE_DIRS}
)

# Link libraries
target_link_libraries(wfc
    ${LV2_LIBRARIES}
    m  # Math library
)

# Install paths
if(NOT DEFINED LV2_INSTALL_DIR)
    set(LV2_INSTALL_DIR "${CMAKE_INSTALL_PREFIX}/lib/lv2" CACHE PATH "LV2 plugin install directory")
endif()

set(BUNDLE_DIR "${LV2_INSTALL_DIR}/wfc.lv2")

# Install plugin bundle
install(TARGETS wfc DESTINATION "${BUNDLE_DIR}")
install(FILES
    manifest.ttl
    wfc.ttl
    DESTINATION "${BUNDLE_DIR}"
)

# Create install script
# configure_file(
#     "${CMAKE_CURRENT_SOURCE_DIR}/install.sh.in"
#     "${CMAKE_CURRENT_BINARY_DIR}/install.sh"
#     @ONLY
# )

# Uninstall target
configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake_uninstall.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake"
    IMMEDIATE @ONLY
)

add_custom_target(uninstall
    COMMAND ${CMAKE_COMMAND} -P ${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake
)

# Package configuration
set(CPACK_PACKAGE_NAME "wfc-lv2")
set(CPACK_PACKAGE_VERSION "${PROJECT_VERSION}")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "${PROJECT_DESCRIPTION}")
set(CPACK_PACKAGE_CONTACT "wfc-audio@example.com")
set(CPACK_GENERATOR "DEB;RPM;TGZ")
set(CPACK_DEBIAN_PACKAGE_DEPENDS "lv2-dev")
set(CPACK_RPM_PACKAGE_REQUIRES "lv2-devel")

include(CPack)

# Print configuration
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "Install prefix: ${CMAKE_INSTALL_PREFIX}")
message(STATUS "LV2 plugin directory: ${BUNDLE_DIR}")
message(STATUS "Compiler flags: ${CMAKE_CXX_FLAGS}")

# Development helpers
add_custom_target(lint
    COMMAND cppcheck --enable=all --std=c++17 --suppress=missingIncludeSystem src/
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMENT "Running static analysis"
)

add_custom_target(format
    COMMAND clang-format -i src/*.cpp src/*.h
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMENT "Formatting source code"
)