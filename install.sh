#!/bin/bash

# WFC LV2 Plugin Build and Install Script
# This script builds the plugin and installs it to ~/.lv2/wfc.lv2/

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}WFC LV2 Plugin Build and Install Script${NC}"
echo "========================================"

# Check if we're in the right directory
if [ ! -f "CMakeLists.txt" ]; then
    echo -e "${RED}Error: CMakeLists.txt not found. Please run this script from the project root directory.${NC}"
    exit 1
fi

# Create build directory if it doesn't exist
if [ ! -d "build" ]; then
    echo -e "${YELLOW}Creating build directory...${NC}"
    mkdir build
fi

# Enter build directory
cd build

# Configure with CMake
echo -e "${YELLOW}Configuring with CMake...${NC}"
cmake ..

# Clean and build
echo -e "${YELLOW}Building plugin...${NC}"
make clean
make -j$(nproc)

# Check if build was successful
if [ ! -f "wfc.so" ]; then
    echo -e "${RED}Error: Build failed - wfc.so not found${NC}"
    exit 1
fi

# Create plugin bundle directory
PLUGIN_DIR="$HOME/.lv2/wfc.lv2"
echo -e "${YELLOW}Creating plugin directory: $PLUGIN_DIR${NC}"
mkdir -p "$PLUGIN_DIR"

# Copy plugin files
echo -e "${YELLOW}Copying plugin files...${NC}"
cp wfc.so "$PLUGIN_DIR/"
cp ../manifest.ttl "$PLUGIN_DIR/"
cp ../wfc.ttl "$PLUGIN_DIR/"

# Verify installation
if [ -f "$PLUGIN_DIR/wfc.so" ] && [ -f "$PLUGIN_DIR/manifest.ttl" ] && [ -f "$PLUGIN_DIR/wfc.ttl" ]; then
    echo -e "${GREEN}Success! Plugin installed to: $PLUGIN_DIR${NC}"
    echo ""
    echo "Plugin files:"
    ls -la "$PLUGIN_DIR"
    echo ""
    echo -e "${YELLOW}Next steps:${NC}"
    echo "1. Restart your DAW (Reaper, Ardour, etc.)"
    echo "2. Rescan plugins if needed"
    echo "3. Look for 'WFC Audio Synthesizer' in your plugin list"
    echo ""
    echo -e "${GREEN}Installation complete!${NC}"
else
    echo -e "${RED}Error: Installation failed - some files were not copied correctly${NC}"
    exit 1
fi