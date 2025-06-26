#include <lv2/core/lv2.h>
#include <lv2/core/lv2_util.h>
#include <lv2/atom/atom.h>
#include <lv2/atom/util.h>
#include <lv2/urid/urid.h>
#include <lv2/log/log.h>
#include <lv2/log/logger.h>

#include <cmath>
#include <cstring>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>
#include <random>
#include <string>

#define WFC_URI "http://purl.org/stuff/wfc-lv2"

// Port indices
typedef enum {
    WFC_INPUT = 0,
    WFC_OUTPUT = 1,
    WFC_PATTERN_SIZE = 2,
    WFC_QUANT_LEVELS = 3,
    WFC_PATTERN_THRESHOLD = 4,
    WFC_MIX = 5,
    WFC_LEARNING_RATE = 6,
    WFC_RANDOMNESS = 7,
    WFC_BUFFER_SIZE = 8,
    WFC_MEMORY_LIMIT = 9
} PortIndex;

// Pattern data structure
struct PatternData {
    std::unordered_set<std::string> nextPatterns;
    int count;
    
    PatternData() : count(0) {}
};

class WaveformWFC {
public:
    WaveformWFC(int patternSize = 8, int quantLevels = 64, size_t maxBufferSize = 1048576)
        : patternSize_(patternSize)
        , quantLevels_(quantLevels)
        , maxBufferSize_(maxBufferSize)
        , currentBufferSize_(0)
        , maxPatterns_(100000)  // Default pattern limit
        , rng_(std::random_device{}()) {
        
        context_.resize(patternSize_ - 1, 0.0f);
        
        // Pre-allocate buffer with maximum size
        learningBuffer_.reserve(maxBufferSize_);
        
        // Pre-allocate temporary buffers to avoid runtime allocation
        tempPattern_.reserve(128);
        tempNextPattern_.reserve(128);
        tempString_.reserve(1024);
        
        // Initialize performance counters
        totalSamplesProcessed_ = 0;
        patternsGenerated_ = 0;
    }
    
    ~WaveformWFC() {
        // Cleanup is automatic with STL containers
    }
    
    void setPatternSize(int size) {
        if (size != patternSize_ && size >= 2 && size <= 128) {  // Increased max size
            patternSize_ = size;
            context_.resize(patternSize_ - 1, 0.0f);
            // Don't clear patterns immediately - let them fade out naturally
        }
    }
    
    void setQuantLevels(int levels) {
        if (levels != quantLevels_ && levels >= 16 && levels <= 1024) {  // Increased max levels
            quantLevels_ = levels;
            // Don't clear patterns immediately for smoother transitions
        }
    }
    
    void setBufferSize(size_t size) {
        // Clamp to valid range and round to nearest power of 2
        size = std::clamp(size, static_cast<size_t>(1024), maxBufferSize_);
        
        // Round to nearest power of 2
        if (size > 0) {
            size_t powerOf2 = 1;
            while (powerOf2 < size) {
                powerOf2 <<= 1;
            }
            // Choose the closer power of 2
            if (powerOf2 > size && (powerOf2 >> 1) > 0) {
                size_t lower = powerOf2 >> 1;
                if (size - lower < powerOf2 - size) {
                    size = lower;
                } else {
                    size = powerOf2;
                }
            } else {
                size = powerOf2;
            }
        }
        
        if (size != currentBufferSize_) {
            currentBufferSize_ = size;
            
            // Resize buffer, preserving recent data if possible
            if (learningBuffer_.size() > size) {
                // Keep the most recent samples
                std::vector<float> temp(learningBuffer_.end() - size, learningBuffer_.end());
                learningBuffer_ = std::move(temp);
                bufferPos_ = 0;
            }
            
            learningBuffer_.resize(size);
        }
    }
    
    void setMemoryLimit(size_t maxPatterns) {
        maxPatterns_ = std::max(static_cast<size_t>(1000), maxPatterns);
    }
    
    void setPatternThreshold(int threshold) {
        patternThreshold_ = std::max(1, threshold);
    }
    
    void setRandomness(float randomness) {
        randomness_ = std::clamp(randomness, 0.0f, 1.0f);
    }
    
    // Get memory usage statistics
    size_t getPatternCount() const { return patterns_.size(); }
    size_t getBufferSize() const { return learningBuffer_.size(); }
    size_t getMemoryUsage() const {
        size_t usage = sizeof(*this);
        usage += learningBuffer_.capacity() * sizeof(float);
        usage += context_.capacity() * sizeof(float);
        
        // Estimate pattern storage size
        for (const auto& [key, data] : patterns_) {
            usage += key.size() + sizeof(data) + data.nextPatterns.size() * 32; // Rough estimate
        }
        
        return usage;
    }
    
    // Quantize sample from [-1, 1] to discrete levels
    int quantize(float sample) {
        sample = std::clamp(sample, -1.0f, 1.0f);
        return static_cast<int>((sample + 1.0f) * quantLevels_ / 2.0f);
    }
    
    // Dequantize level back to [-1, 1]
    float dequantize(int level) {
        return (static_cast<float>(level) / (quantLevels_ - 1)) * 2.0f - 1.0f;
    }
    
    // Learn patterns from input samples with advanced memory management
    void learnFromSamples(const float* samples, size_t count, float learningRate) {
        if (count < patternSize_ || learningRate <= 0.0f) return;
        
        // Limit processing for real-time performance
        size_t maxSamplesToProcess = std::min(count, static_cast<size_t>(128)); // Much smaller chunks
        size_t learnStep = learningRate > 0.5f ? 1 : 2; // Adaptive step size
        
        // Process samples in small chunks to maintain real-time performance
        size_t samplesProcessed = 0;
        
        for (size_t i = 0; i < count - patternSize_ && samplesProcessed < maxSamplesToProcess; i += learnStep) {
            // Use pre-allocated buffer instead of creating new vector
            tempPattern_.clear();
            
            // Create pattern from current window
            for (int j = 0; j < patternSize_; ++j) {
                tempPattern_.push_back(quantize(samples[i + j]));
            }
            
            std::string patternKey = vectorToString(tempPattern_);
            
            // Check memory limits before adding new patterns
            if (patterns_.find(patternKey) == patterns_.end() && patterns_.size() >= maxPatterns_) {
                // Memory limit reached - remove least frequent patterns
                prunePatterns(0.1f); // Remove bottom 10%
            }
            
            patterns_[patternKey].count++;
            
            // Add next pattern if available
            if (i + patternSize_ < count) {
                // Use pre-allocated buffer for next pattern too
                tempNextPattern_.clear();
                
                for (int j = 1; j <= patternSize_; ++j) {
                    tempNextPattern_.push_back(quantize(samples[i + j]));
                }
                
                std::string nextKey = vectorToString(tempNextPattern_);
                patterns_[patternKey].nextPatterns.insert(nextKey);
            }
            
            samplesProcessed++;
        }
        
        totalSamplesProcessed_ += samplesProcessed;
        
        // Periodic cleanup based on pattern threshold (much less frequent)
        if (totalSamplesProcessed_ % 500000 == 0) {
            prunePatterns();
        }
    }
    
    // Add sample to learning buffer
    void addToBuffer(float sample) {
        if (learningBuffer_.empty()) return;
        
        learningBuffer_[bufferPos_] = sample;
        bufferPos_ = (bufferPos_ + 1) % learningBuffer_.size();
    }
    
    // Learn from current buffer contents
    void learnFromBuffer(float learningRate) {
        if (!learningBuffer_.empty()) {
            learnFromSamples(learningBuffer_.data(), learningBuffer_.size(), learningRate);
        }
    }
    
    // Generate next sample based on current context
    float generateSample() {
        if (patterns_.empty()) {
            // No patterns learned yet, return gentle noise
            std::uniform_real_distribution<float> dist(-0.1f, 0.1f);
            return dist(rng_);
        }
        
        // Convert current context to quantized pattern
        std::vector<int> contextPattern;
        contextPattern.reserve(context_.size());
        for (float sample : context_) {
            contextPattern.push_back(quantize(sample));
        }
        
        std::string contextKey = vectorToString(contextPattern);
        
        // Find matching patterns
        std::vector<int> candidates;
        int totalWeight = 0;
        
        for (const auto& [pattern, data] : patterns_) {
            std::vector<int> patternVec = stringToVector(pattern);
            
            // Check if pattern starts with our context
            bool matches = true;
            for (size_t i = 0; i < contextPattern.size() && i < patternVec.size() - 1; ++i) {
                if (contextPattern[i] != patternVec[i]) {
                    matches = false;
                    break;
                }
            }
            
            if (matches && !patternVec.empty()) {
                int weight = data.count;
                for (int i = 0; i < weight; ++i) {
                    candidates.push_back(patternVec.back());
                }
                totalWeight += weight;
            }
        }
        
        float nextSample;
        
        if (candidates.empty() || (randomness_ > 0.0f && 
            std::uniform_real_distribution<float>(0.0f, 1.0f)(rng_) < randomness_)) {
            // No candidates or random mode - generate random sample
            std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
            nextSample = dist(rng_);
        } else {
            // Select weighted random candidate
            std::uniform_int_distribution<size_t> dist(0, candidates.size() - 1);
            int selectedLevel = candidates[dist(rng_)];
            nextSample = dequantize(selectedLevel);
        }
        
        // Update context
        context_.erase(context_.begin());
        context_.push_back(nextSample);
        
        return nextSample;
    }
    
private:
    int patternSize_;
    int quantLevels_;
    int patternThreshold_ = 2;
    float randomness_ = 0.1f;
    
    // Buffer management
    size_t maxBufferSize_;
    size_t currentBufferSize_;
    size_t maxPatterns_;
    std::vector<float> learningBuffer_;
    size_t bufferPos_ = 0;
    
    // Performance counters
    size_t totalSamplesProcessed_;
    size_t patternsGenerated_;
    
    // Pre-allocated buffers to avoid runtime allocation
    mutable std::vector<int> tempPattern_;
    mutable std::vector<int> tempNextPattern_;
    mutable std::string tempString_;
    
    std::vector<float> context_;
    std::unordered_map<std::string, PatternData> patterns_;
    std::mt19937 rng_;
    
    std::string vectorToString(const std::vector<int>& vec) {
        std::string result;
        for (size_t i = 0; i < vec.size(); ++i) {
            if (i > 0) result += ",";
            result += std::to_string(vec[i]);
        }
        return result;
    }
    
    std::vector<int> stringToVector(const std::string& str) {
        std::vector<int> result;
        size_t start = 0;
        size_t end = 0;
        
        while ((end = str.find(',', start)) != std::string::npos) {
            result.push_back(std::stoi(str.substr(start, end - start)));
            start = end + 1;
        }
        result.push_back(std::stoi(str.substr(start)));
        
        return result;
    }
    
    // Remove patterns with low frequency
    void prunePatterns(float threshold = 0.0f) {
        if (patterns_.empty()) return;
        
        if (threshold > 0.0f) {
            // Remove bottom percentage
            std::vector<std::pair<std::string, int>> patternCounts;
            for (const auto& [key, data] : patterns_) {
                patternCounts.push_back({key, data.count});
            }
            
            std::sort(patternCounts.begin(), patternCounts.end(), 
                     [](const auto& a, const auto& b) { return a.second < b.second; });
            
            size_t removeCount = static_cast<size_t>(patternCounts.size() * threshold);
            for (size_t i = 0; i < removeCount; ++i) {
                patterns_.erase(patternCounts[i].first);
            }
        } else {
            // Remove patterns below threshold
            auto it = patterns_.begin();
            while (it != patterns_.end()) {
                if (it->second.count < patternThreshold_) {
                    it = patterns_.erase(it);
                } else {
                    ++it;
                }
            }
        }
    }
};

// Plugin instance
typedef struct {
    // Ports
    const float* input;
    float* output;
    const float* patternSize;
    const float* quantLevels;
    const float* patternThreshold;
    const float* mix;
    const float* learningRate;
    const float* randomness;
    const float* bufferSize;
    const float* memoryLimit;
    
    // Internal state
    WaveformWFC* wfc;
    std::vector<float> inputBuffer;
    size_t bufferPos;
    size_t learningCounter;
    size_t lastBufferSize;
    
    // Sample rate
    double sampleRate;
    
    // Performance monitoring
    size_t totalSamplesProcessed;
    size_t lastMemoryUsage;
} WfcPlugin;

static LV2_Handle instantiate(const LV2_Descriptor* descriptor,
                            double rate,
                            const char* bundle_path,
                            const LV2_Feature* const* features) {
    
    WfcPlugin* plugin = new WfcPlugin;
    
    // Initialize with large buffer support (1MB default)
    plugin->wfc = new WaveformWFC(8, 64, 1048576);
    plugin->sampleRate = rate;
    plugin->bufferPos = 0;
    plugin->learningCounter = 0;
    plugin->lastBufferSize = 4096; // Default buffer size (power of 2)
    plugin->totalSamplesProcessed = 0;
    plugin->lastMemoryUsage = 0;
    
    // Initialize input buffer
    plugin->inputBuffer.resize(plugin->lastBufferSize);
    
    // Set initial buffer size
    plugin->wfc->setBufferSize(plugin->lastBufferSize);
    
    return (LV2_Handle)plugin;
}

static void connect_port(LV2_Handle instance, uint32_t port, void* data) {
    WfcPlugin* plugin = (WfcPlugin*)instance;
    
    switch ((PortIndex)port) {
        case WFC_INPUT:
            plugin->input = (const float*)data;
            break;
        case WFC_OUTPUT:
            plugin->output = (float*)data;
            break;
        case WFC_PATTERN_SIZE:
            plugin->patternSize = (const float*)data;
            break;
        case WFC_QUANT_LEVELS:
            plugin->quantLevels = (const float*)data;
            break;
        case WFC_PATTERN_THRESHOLD:
            plugin->patternThreshold = (const float*)data;
            break;
        case WFC_MIX:
            plugin->mix = (const float*)data;
            break;
        case WFC_LEARNING_RATE:
            plugin->learningRate = (const float*)data;
            break;
        case WFC_RANDOMNESS:
            plugin->randomness = (const float*)data;
            break;
        case WFC_BUFFER_SIZE:
            plugin->bufferSize = (const float*)data;
            break;
        case WFC_MEMORY_LIMIT:
            plugin->memoryLimit = (const float*)data;
            break;
    }
}

static void run(LV2_Handle instance, uint32_t nFrames) {
    WfcPlugin* plugin = (WfcPlugin*)instance;
    
    // Update parameters (less frequently to avoid audio thread issues)
    static uint32_t paramUpdateCounter = 0;
    if (++paramUpdateCounter >= 1024) { // Update every ~20ms at 48kHz
        paramUpdateCounter = 0;
        plugin->wfc->setPatternSize((int)*plugin->patternSize);
        plugin->wfc->setQuantLevels((int)*plugin->quantLevels);
        plugin->wfc->setPatternThreshold((int)*plugin->patternThreshold);
        plugin->wfc->setRandomness(*plugin->randomness);
        
        // Handle buffer size changes
        size_t newBufferSize = (size_t)*plugin->bufferSize;
        if (newBufferSize != plugin->lastBufferSize) {
            plugin->lastBufferSize = newBufferSize;
            plugin->wfc->setBufferSize(newBufferSize);
            plugin->inputBuffer.resize(newBufferSize);
            plugin->bufferPos = 0; // Reset position after resize
        }
    }
    
    float mixAmount = std::clamp(*plugin->mix, 0.0f, 1.0f);
    float learningRate = std::clamp(*plugin->learningRate, 0.0f, 1.0f);
    
    for (uint32_t i = 0; i < nFrames; ++i) {
        float inputSample = plugin->input[i];
        
        // Add to learning buffer
        plugin->inputBuffer[plugin->bufferPos] = inputSample;
        plugin->bufferPos = (plugin->bufferPos + 1) % plugin->inputBuffer.size();
        
        // Real-time learning from buffer (every 64 samples for better responsiveness)
        if (++plugin->learningCounter >= 64) {
            plugin->learningCounter = 0;
            // Learn from smaller chunks to reduce processing spikes
            size_t chunkSize = std::min(plugin->inputBuffer.size(), static_cast<size_t>(512));
            plugin->wfc->learnFromSamples(plugin->inputBuffer.data(), 
                                        chunkSize, 
                                        learningRate);
        }
        
        // Generate WFC sample
        float wfcSample = plugin->wfc->generateSample();
        
        // Mix with original
        plugin->output[i] = inputSample * (1.0f - mixAmount) + wfcSample * mixAmount;
    }
}

static void cleanup(LV2_Handle instance) {
    WfcPlugin* plugin = (WfcPlugin*)instance;
    delete plugin->wfc;
    delete plugin;
}

static const LV2_Descriptor descriptor = {
    WFC_URI,
    instantiate,
    connect_port,
    nullptr, // activate
    run,
    nullptr, // deactivate
    cleanup,
    nullptr  // extension_data
};

LV2_SYMBOL_EXPORT const LV2_Descriptor* lv2_descriptor(uint32_t index) {
    switch (index) {
        case 0:
            return &descriptor;
        default:
            return nullptr;
    }
}