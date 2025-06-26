# WFC Audio Synthesizer - LV2 Plugin

A real-time audio plugin that applies Wave Function Collapse (WFC) algorithm to audio synthesis and transformation. This plugin learns patterns from incoming audio and generates new audio content based on those patterns, creating unique variations while preserving the structural characteristics of the input.

## Algorithm Overview

### Wave Function Collapse for Audio

Wave Function Collapse is a constraint-solving algorithm originally developed for procedural content generation in games. This plugin adapts the algorithm for audio processing by treating audio waveforms as a temporal grid where:

- **X-axis**: Represents time (sequential audio samples)
- **Y-axis**: Represents quantized amplitude levels
- **Patterns**: Short sequences of amplitude values that capture local audio characteristics

### How It Works

1. **Pattern Extraction**: The algorithm continuously analyzes the input audio stream, extracting short overlapping patterns (typically 2-32 samples long). Each pattern represents a small fragment of the audio's local structure.

2. **Quantization**: Audio samples are quantized into discrete amplitude levels (16-256 levels) to create a finite state space for pattern matching.

3. **Constraint Learning**: The plugin builds a database of valid pattern transitions by observing which patterns naturally follow each other in the input audio.

4. **Real-time Generation**: For each output sample, the algorithm:
   - Examines the current context (recent samples)
   - Finds all learned patterns that match this context
   - Selects the next sample based on weighted probability from valid continuations
   - Updates the context with the new sample

5. **Coherence Maintenance**: By using overlapping contexts and weighted pattern selection, the output maintains both local coherence (smooth transitions) and global structure (statistical properties of the original).

### Large Buffer Support

The plugin now supports much larger learning buffers (up to 1 million samples), enabling:

- **Long-term Pattern Recognition**: Capture patterns that span several seconds or minutes
- **Complex Musical Structure Learning**: Understand chord progressions, melodic phrases, and rhythmic cycles
- **Extended Memory**: Store and recall patterns from much longer audio contexts
- **Adaptive Memory Management**: Automatically prune less important patterns to maintain performance

#### Buffer Size Recommendations (Powers of 2):

- **1,024 samples (~23ms)**: Minimum buffer for basic pattern detection
- **4,096 samples (~93ms)**: Default - good balance of latency and pattern recognition
- **8,192 samples (~186ms)**: Better for rhythmic patterns and short melodic phrases
- **16,384 samples (~372ms)**: Captures longer musical phrases and transitions
- **65,536 samples (~1.5 seconds)**: Learns complex musical structures and progressions
- **262,144 samples (~6 seconds)**: Extended pattern recognition for complex compositions
- **1,048,576 samples (~24 seconds)**: Maximum buffer for learning very long structures

#### Memory Management:

The plugin automatically manages memory usage through:
- **Pattern Pruning**: Removes infrequently used patterns when memory limits are reached
- **Frequency-based Prioritization**: Keeps the most commonly occurring patterns
- **Adaptive Learning**: Reduces processing overhead for very large buffers
- **Real-time Monitoring**: Tracks memory usage and pattern count

**Note**: Larger buffers require more RAM and CPU. Monitor system performance when using buffers above 100,000 samples.

### Key Features

- **Adaptive Learning**: Continuously learns from input audio in real-time
- **Pattern Memory**: Maintains a dynamic database of audio patterns and their relationships  
- **Constraint Satisfaction**: Ensures generated audio follows learned structural rules
- **Statistical Preservation**: Maintains amplitude distributions and frequency characteristics
- **Real-time Performance**: Optimized for low-latency audio processing
- **Large Buffer Support**: Handle buffers up to 1 million samples for extended pattern recognition
- **Smart Memory Management**: Automatic pattern pruning and memory optimization

## Parameters

| Parameter | Range | Default | Description |
|-----------|-------|---------|-------------|
| **Pattern Size** | 2-128 | 8 | Length of patterns in samples. Smaller = more chaotic, larger = more structured |
| **Quantization Levels** | 16-1024 | 64 | Number of amplitude levels. More levels = higher fidelity but more memory |
| **Pattern Threshold** | 1-10 | 2 | Minimum occurrences needed to keep a pattern. Higher = more selective |
| **Mix** | 0.0-1.0 | 0.5 | Blend between original (0.0) and WFC-generated (1.0) audio |
| **Learning Rate** | 0.0-1.0 | 0.1 | How quickly new patterns are learned. Higher = faster adaptation |
| **Randomness** | 0.0-1.0 | 0.1 | Amount of randomness in pattern selection. Higher = more variation |
| **Buffer Size** | 1024-1048576 | 4096 | Learning buffer size in samples (powers of 2, ~23ms to 23.7s at 44.1kHz) |
| **Memory Limit** | 1000-1000000 | 100000 | Maximum number of patterns stored in memory before pruning occurs |

## Build Instructions

### Prerequisites

#### Ubuntu/Debian:
```bash
sudo apt-get update
sudo apt-get install build-essential cmake pkg-config lv2-dev
```

#### Fedora/RHEL:
```bash
sudo dnf install gcc-c++ cmake pkgconfig lv2-devel
```

#### Arch Linux:
```bash
sudo pacman -S base-devel cmake pkgconf lv2
```

### Quick Install

For a quick build and install, use the provided script:

```bash
./install.sh
```

This script will automatically build the plugin and install it to `~/.lv2/wfc.lv2/`.

### Manual Building

1. **Clone or download the plugin source code**
2. **Create build directory:**
   ```bash
   mkdir build
   cd build
   ```

3. **Configure with CMake:**
   ```bash
   cmake ..
   ```
   
   For debug build:
   ```bash
   cmake -DCMAKE_BUILD_TYPE=Debug ..
   ```

4. **Build the plugin:**
   ```bash
   make -j$(nproc)
   ```
   
   For a clean rebuild:
   ```bash
   make clean && make -j$(nproc)
   ```

5. **Install the plugin manually:**
   ```bash
   # Create the plugin bundle directory
   mkdir -p ~/.lv2/wfc.lv2
   
   # Copy the plugin files
   cp wfc.so ~/.lv2/wfc.lv2/
   cp ../manifest.ttl ../wfc.ttl ~/.lv2/wfc.lv2/
   ```
   
   The plugin will be installed to `~/.lv2/wfc.lv2/` in your user directory.

### Verification

After installation, verify the plugin is available:
```bash
lv2ls | grep wfc
```

You should see: `http://purl.org/stuff/wfc-lv2`

### Manual Installation for DAWs (Reaper, etc.)

If you prefer to install manually or your DAW doesn't scan the system LV2 directories:

1. **Create the plugin bundle directory:**
   ```bash
   mkdir -p ~/.lv2/wfc.lv2
   ```

2. **Copy the plugin files:**
   ```bash
   # From your build directory
   cp wfc.so ~/.lv2/wfc.lv2/
   cp ../manifest.ttl ../wfc.ttl ~/.lv2/wfc.lv2/
   ```

3. **Restart your DAW and rescan plugins:**
   - **Reaper**: Options → Preferences → Plug-ins → Re-scan
   - **Other DAWs**: Look for plugin scan/refresh option in preferences

4. **Find the plugin:**
   - Look for "WFC" or "Wave Function Collapse" in your plugin list
   - The plugin should appear in the LV2 or Effects category

5. **For plugin updates:**
   - After rebuilding, copy the new `wfc.so` file: `cp wfc.so ~/.lv2/wfc.lv2/`
   - Restart your DAW to load the updated plugin

The plugin bundle contains:
- `wfc.so` - the compiled plugin binary
- `manifest.ttl` - plugin manifest file  
- `wfc.ttl` - plugin description and parameters

## Usage

### Loading the Plugin

The plugin can be loaded in any LV2-compatible host:

- **Ardour**: Add as an insert effect on an audio track
- **Carla**: Load as an LV2 plugin
- **Qtractor**: Insert on track or bus
- **Reaper**: Load via LV2 wrapper
- **Command Line**: Use `jalv` for testing

### Example with jalv:
```bash
jalv http://purl.org/stuff/wfc-lv2
```

### Typical Usage Scenarios

1. **Ambient Texture Generation**: Set high mix level, medium pattern size, low randomness
2. **Rhythmic Variation**: Small pattern size, high learning rate, medium randomness
3. **Harmonic Exploration**: Large pattern size, low learning rate, high quantization
4. **Glitch Effects**: Small pattern size, high randomness, low pattern threshold

### Tips for Best Results

- **Start with low mix values** (0.1-0.3) to hear the effect subtly
- **Adjust pattern size** based on input material complexity
- **Use higher learning rates** for rapidly changing input
- **Increase randomness** for more experimental results
- **Monitor CPU usage** - complex patterns may require more processing

#### Buffer Size Usage Tips:
- **For low latency**: Use 1,024-4,096 samples (23-93ms)
- **For rhythmic material**: Use 4,096-8,192 samples (93-186ms)
- **For melodic content**: Use 8,192-16,384 samples (186-372ms)  
- **For complex compositions**: Use 16,384+ samples (372ms+)
- **Start with 4,096 samples** (default) and adjust based on material complexity
- **Watch CPU usage** - larger buffers require more processing
- **Use pattern threshold 3-5** with large buffers to reduce noise

## Technical Details

### Memory Usage

The plugin's memory usage scales with buffer size and pattern complexity:

**Buffer Memory**:
- 1,024 samples: ~4 KB
- 44,100 samples: ~176 KB  
- 441,000 samples: ~1.7 MB
- 1,048,576 samples: ~4.2 MB

**Pattern Database**:
- 1,000 patterns: ~50-200 KB
- 10,000 patterns: ~500 KB - 2 MB
- 100,000 patterns: ~5-20 MB
- 1,000,000 patterns: ~50-200 MB

**Total Memory Usage**: Typically 1-50 MB for normal operation, can reach 200+ MB with maximum settings.

**Memory Optimization**:
- Pattern pruning removes infrequent patterns automatically
- Configurable memory limits prevent unbounded growth
- Efficient hash-based pattern storage
- Lazy allocation of pattern data structures

### CPU Performance

The plugin is optimized for real-time performance:
- Pattern lookup using hash tables
- Efficient quantization/dequantization
- Adaptive learning to prevent database bloat
- SIMD optimization opportunities for future versions

### Latency

The plugin introduces minimal latency:
- Algorithmic latency: 0 samples
- Processing latency: Depends on pattern complexity
- Typical latency: < 1ms on modern hardware

## Troubleshooting

### Build Issues

**CMake not finding LV2:**
```bash
# Install lv2-dev/lv2-devel package
# Or specify path manually:
cmake -DLV2_INCLUDE_DIRS=/path/to/lv2/include ..
```

**Compilation errors:**
- Ensure C++17 support (GCC 7+ or Clang 5+)
- Check all dependencies are installed
- Try debug build for more verbose errors

### Runtime Issues

**Plugin not appearing in host:**
- Verify installation path matches host's LV2 path
- Check file permissions
- Use `lv2ls` to verify plugin is detected

**Audio glitches or artifacts:**
- Reduce pattern size or quantization levels
- Lower learning rate
- Increase pattern threshold
- Check CPU usage
- **For large buffers**: Reduce buffer size or memory limit

**High memory usage:**
- Reduce buffer size or memory limit
- Increase pattern threshold to be more selective
- Lower quantization levels
- Restart plugin to clear accumulated patterns

**Slow response with large buffers:**
- Reduce learning rate for large buffers
- Increase pattern threshold
- Use smaller quantization levels
- Monitor system resources

**No audio output:**
- Verify mix parameter is > 0
- Check that patterns are being learned (may take time)
- Try increasing randomness for initial output

## Development

### Project Structure
```
wfc-lv2/
├── src/
│   └── wfc.cpp          # Main plugin implementation
├── manifest.ttl         # LV2 manifest
├── wfc.ttl             # Plugin descriptor
├── CMakeLists.txt       # Build configuration
└── README.md           # This file
```

### Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes with appropriate tests
4. Submit a pull request

### Code Style

- C++17 standard
- RAII patterns
- Minimal external dependencies
- Real-time safe operations only

## License

This project is licensed under the MIT License. See LICENSE file for details.

## Acknowledgments

- Inspired by the Wave Function Collapse algorithm by Maxim Gumin
- Built on the LV2 plugin standard
- Thanks to the open-source audio development community

## Support

For issues, feature requests, or questions:
- Open an issue on the project repository
- Join the discussion in audio development forums
- Check the LV2 community resources

---

*The WFC Audio Synthesizer brings the power of constraint-based procedural generation to audio processing, opening new possibilities for creative sound design and algorithmic composition.*