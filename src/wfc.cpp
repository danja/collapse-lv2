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
    WFC_MEMORY_LIMIT = 9,
    WFC_PERIOD_DETECT = 10,
    WFC_MIN_FREQ = 11,
    WFC_MAX_FREQ = 12,
    WFC_PATTERN_REPEATS = 13
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
        contextPattern_.reserve(128);
        candidates_.reserve(10000);  // Large enough for typical use
        patternVec_.reserve(128);
        
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
        return static_cast<int>((sample + 1.0f) * (quantLevels_ - 1) / 2.0f);
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
            
            // Add next pattern if available and within memory limits
            if (i + patternSize_ < count && patterns_[patternKey].nextPatterns.size() < 100) {
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
    
    // Update context with new input sample using circular buffer
    void updateContext(float sample) {
        if (!context_.empty()) {
            context_[contextIndex_] = sample;
            contextIndex_ = (contextIndex_ + 1) % context_.size();
        }
    }
    
    // Generate next sample based on current context
    float generateSample() {
        // Don't generate until we have enough patterns
        if (patterns_.empty() || totalSamplesProcessed_ < 1000) {
            return 0.0f; // Return silence until we learn patterns
        }
        
        // Convert current context to quantized pattern using pre-allocated buffer
        contextPattern_.clear();
        // Read from circular buffer in correct order
        for (size_t i = 0; i < context_.size(); ++i) {
            size_t idx = (contextIndex_ + i) % context_.size();
            contextPattern_.push_back(quantize(context_[idx]));
        }
        
        std::string contextKey = vectorToString(contextPattern_);
        
        // Find matching patterns using pre-allocated buffer
        candidates_.clear();
        int totalWeight = 0;
        
        for (const auto& [pattern, data] : patterns_) {
            stringToVector(pattern, patternVec_);  // Use pre-allocated buffer
            
            // Check if pattern starts with our context
            bool matches = true;
            for (size_t i = 0; i < contextPattern_.size() && i < patternVec_.size() - 1; ++i) {
                if (contextPattern_[i] != patternVec_[i]) {
                    matches = false;
                    break;
                }
            }
            
            if (matches && !patternVec_.empty()) {
                int weight = data.count;
                for (int i = 0; i < weight; ++i) {
                    candidates_.push_back(patternVec_.back());
                }
                totalWeight += weight;
            }
        }
        
        float nextSample;
        
        if (candidates_.empty() || (randomness_ > 0.0f && 
            std::uniform_real_distribution<float>(0.0f, 1.0f)(rng_) < randomness_)) {
            // No candidates or random mode - generate quiet random sample
            std::uniform_real_distribution<float> dist(-0.01f, 0.01f);
            nextSample = dist(rng_);
        } else {
            // Select weighted random candidate
            std::uniform_int_distribution<size_t> dist(0, candidates_.size() - 1);
            int selectedLevel = candidates_[dist(rng_)];
            nextSample = dequantize(selectedLevel);
        }
        
        // Update context using circular buffer
        updateContext(nextSample);
        
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
    mutable std::vector<int> contextPattern_;
    mutable std::vector<int> candidates_;
    mutable std::vector<int> patternVec_;
    
    std::vector<float> context_;
    size_t contextIndex_ = 0;  // For circular context buffer
    std::unordered_map<std::string, PatternData> patterns_;
    std::mt19937 rng_;
    
    std::string vectorToString(const std::vector<int>& vec) const {
        tempString_.clear();
        for (size_t i = 0; i < vec.size(); ++i) {
            if (i > 0) tempString_ += ",";
            tempString_ += std::to_string(vec[i]);
        }
        return tempString_;
    }
    
    void stringToVector(const std::string& str, std::vector<int>& result) const {
        result.clear();
        size_t start = 0;
        size_t end = 0;
        
        while ((end = str.find(',', start)) != std::string::npos) {
            result.push_back(std::stoi(str.substr(start, end - start)));
            start = end + 1;
        }
        result.push_back(std::stoi(str.substr(start)));
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
    const float* periodDetectEnable;
    const float* minFreq;
    const float* maxFreq;
    const float* patternRepeats;
    
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
    
    // Period detection state
    std::vector<float> correlationBuffer;
    size_t correlationPos;
    size_t detectedPeriod;
    float correlationThreshold;
    
    // Pattern alternation state
    std::vector<float> currentPattern;  // Current predicted pattern
    std::vector<float> nextPattern;     // Next predicted pattern (background)
    size_t patternPlaybackPos;          // Position in current pattern
    size_t currentRepeatCount;          // How many times current pattern has played
    bool backgroundPredictionReady;     // Is next pattern ready?
} WfcPlugin;

// Period detection using autocorrelation
static size_t detectPeriod(WfcPlugin* plugin, float minFreq, float maxFreq) {
    // Safety checks
    if (plugin->correlationBuffer.empty() || plugin->correlationPos < 1000) {
        return 0; // Need at least 1000 samples for reliable detection
    }
    
    size_t minPeriod = static_cast<size_t>(plugin->sampleRate / maxFreq);
    size_t maxPeriod = static_cast<size_t>(plugin->sampleRate / minFreq);
    
    // Clamp to reasonable bounds
    minPeriod = std::max(minPeriod, static_cast<size_t>(4));
    maxPeriod = std::min(maxPeriod, plugin->correlationBuffer.size() / 4); // More conservative
    
    if (minPeriod >= maxPeriod) return 0;
    
    float bestCorrelation = 0.0f;
    size_t bestPeriod = 0;
    size_t bufferSize = plugin->correlationBuffer.size();
    
    // Test different periods
    for (size_t period = minPeriod; period <= maxPeriod; period++) {
        float correlation = 0.0f;
        float norm1 = 0.0f, norm2 = 0.0f;
        
        // Calculate correlation for this period
        size_t compareLength = std::min(period * 4, bufferSize - period); // Compare up to 4 cycles
        
        for (size_t i = 0; i < compareLength; i++) {
            size_t idx1 = (plugin->correlationPos + bufferSize - compareLength + i) % bufferSize;
            size_t idx2 = (idx1 + bufferSize - period) % bufferSize;
            
            float val1 = plugin->correlationBuffer[idx1];
            float val2 = plugin->correlationBuffer[idx2];
            
            correlation += val1 * val2;
            norm1 += val1 * val1;
            norm2 += val2 * val2;
        }
        
        // Normalize correlation coefficient
        if (norm1 > 0.0f && norm2 > 0.0f) {
            correlation /= std::sqrt(norm1 * norm2);
        }
        
        if (correlation > bestCorrelation && correlation > plugin->correlationThreshold) {
            bestCorrelation = correlation;
            bestPeriod = period;
        }
    }
    
    return bestPeriod;
}

// Generate a predicted pattern in the background
static void generatePredictedPattern(WfcPlugin* plugin, std::vector<float>& targetPattern, size_t patternLength) {
    targetPattern.clear();
    
    // Safety checks to prevent crashes
    if (patternLength == 0 || patternLength > 10000 || 
        plugin->inputBuffer.size() < 64 || plugin->bufferPos < 32) {
        // Not enough data or invalid pattern length, fill with silence
        targetPattern.resize(std::min(patternLength, static_cast<size_t>(2048)), 0.0f);
        return;
    }
    
    targetPattern.reserve(patternLength);
    
    size_t bufferSize = plugin->inputBuffer.size();
    
    for (size_t i = 0; i < patternLength; i++) {
        // Create a small pattern from recent samples for prediction
        std::vector<float> recentPattern;
        recentPattern.reserve(8);
        
        // Get last 8 samples as pattern
        for (int j = 7; j >= 0; j--) {
            size_t patternPos = (plugin->bufferPos + bufferSize - j - 1 + i) % bufferSize;
            recentPattern.push_back(plugin->inputBuffer[patternPos]);
        }
        
        // Simple WFC: find best matching pattern in buffer and predict next sample
        float bestMatch = 0.0f;
        float minDistance = std::numeric_limits<float>::max();
        
        // Search through buffer for similar patterns (limited search for real-time)
        size_t searchLimit = std::min(bufferSize / 2, static_cast<size_t>(512)); // Smaller search for background
        for (size_t k = 8; k < searchLimit; k += 8) { // Larger step for performance
            float distance = 0.0f;
            
            // Compare pattern
            for (int p = 0; p < 8; p++) {
                size_t comparePos = (k - 8 + p + bufferSize) % bufferSize;
                float diff = recentPattern[p] - plugin->inputBuffer[comparePos];
                distance += diff * diff;
            }
            
            if (distance < minDistance) {
                minDistance = distance;
                // Get the sample that followed this pattern
                size_t nextPos = (k + bufferSize) % bufferSize;
                bestMatch = plugin->inputBuffer[nextPos];
            }
        }
        
        // Clamp predicted sample to prevent overload
        bestMatch = std::clamp(bestMatch, -1.0f, 1.0f);
        targetPattern.push_back(bestMatch);
    }
}

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
    
    // Initialize all port pointers to null for safety
    plugin->input = nullptr;
    plugin->output = nullptr;
    plugin->patternSize = nullptr;
    plugin->quantLevels = nullptr;
    plugin->patternThreshold = nullptr;
    plugin->mix = nullptr;
    plugin->learningRate = nullptr;
    plugin->randomness = nullptr;
    plugin->bufferSize = nullptr;
    plugin->memoryLimit = nullptr;
    plugin->periodDetectEnable = nullptr;
    plugin->minFreq = nullptr;
    plugin->maxFreq = nullptr;
    plugin->patternRepeats = nullptr;
    
    // Initialize period detection - smaller buffer to avoid memory issues
    size_t correlationSize = std::min(static_cast<size_t>(rate * 1.0), static_cast<size_t>(44100)); // Max 1 second
    plugin->correlationBuffer.resize(correlationSize, 0.0f);
    plugin->correlationPos = 0;
    plugin->detectedPeriod = 0;
    plugin->correlationThreshold = 0.7f;
    
    // Initialize pattern alternation
    plugin->currentPattern.clear();
    plugin->nextPattern.clear();
    plugin->patternPlaybackPos = 0;
    plugin->currentRepeatCount = 0;
    plugin->backgroundPredictionReady = false;
    
    // Initialize input buffer with silence
    plugin->inputBuffer.resize(plugin->lastBufferSize, 0.0f);
    
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
        case WFC_PERIOD_DETECT:
            plugin->periodDetectEnable = (const float*)data;
            break;
        case WFC_MIN_FREQ:
            plugin->minFreq = (const float*)data;
            break;
        case WFC_MAX_FREQ:
            plugin->maxFreq = (const float*)data;
            break;
        case WFC_PATTERN_REPEATS:
            plugin->patternRepeats = (const float*)data;
            break;
    }
}

static void run(LV2_Handle instance, uint32_t nFrames) {
    WfcPlugin* plugin = (WfcPlugin*)instance;
    
    // Safety checks to prevent crashes
    if (!plugin || !plugin->input || !plugin->output) {
        return;
    }
    
    // If critical ports aren't connected yet, pass through input
    if (!plugin->mix) {
        for (uint32_t i = 0; i < nFrames; ++i) {
            plugin->output[i] = plugin->input[i]; // Pass through input
        }
        return;
    }
    
    // Update parameters (less frequently to avoid audio thread issues)
    static uint32_t paramUpdateCounter = 0;
    if (++paramUpdateCounter >= 1024) { // Update every ~20ms at 48kHz
        paramUpdateCounter = 0;
        plugin->wfc->setPatternSize((int)*plugin->patternSize);
        plugin->wfc->setQuantLevels((int)*plugin->quantLevels);
        plugin->wfc->setPatternThreshold((int)*plugin->patternThreshold);
        plugin->wfc->setRandomness(*plugin->randomness);
        plugin->wfc->setMemoryLimit((size_t)*plugin->memoryLimit);
        
        // Period detection and buffer size determination
        size_t newBufferSize;
        bool periodDetectEnabled = *plugin->periodDetectEnable > 0.5f;
        if (periodDetectEnabled && plugin->correlationPos > 10000) { // Wait for enough data
            size_t detectedPeriod = detectPeriod(plugin, *plugin->minFreq, *plugin->maxFreq);
            if (detectedPeriod > 0 && detectedPeriod < 100000) { // Sanity check
                // Use detected period as buffer size, round to power of 2
                size_t periodBufferSize = detectedPeriod * 2; // Use 2 cycles for better pattern matching
                
                // Round to nearest power of 2
                size_t powerOf2 = 1;
                while (powerOf2 < periodBufferSize && powerOf2 < 1048576) {
                    powerOf2 <<= 1;
                }
                
                plugin->detectedPeriod = detectedPeriod;
                newBufferSize = powerOf2;
            } else {
                newBufferSize = (size_t)*plugin->bufferSize; // Fallback to manual setting
            }
        } else {
            newBufferSize = (size_t)*plugin->bufferSize; // Manual buffer size
        }
        
        // Handle buffer size changes
        if (newBufferSize != plugin->lastBufferSize) {
            plugin->lastBufferSize = newBufferSize;
            plugin->wfc->setBufferSize(newBufferSize);
            plugin->inputBuffer.resize(newBufferSize, 0.0f); // Initialize with silence
            plugin->bufferPos = 0; // Reset position after resize
        }
    }
    
    // Parameters are used inside the loop, no need to pre-calculate
    
    for (uint32_t i = 0; i < nFrames; ++i) {
        float inputSample = plugin->input[i];
        
        // Add to correlation buffer for period detection
        plugin->correlationBuffer[plugin->correlationPos] = inputSample;
        plugin->correlationPos = (plugin->correlationPos + 1) % plugin->correlationBuffer.size();
        
        // Add to learning buffer
        plugin->inputBuffer[plugin->bufferPos] = inputSample;
        plugin->bufferPos = (plugin->bufferPos + 1) % plugin->inputBuffer.size();
        
        // Variable delay based on buffer size parameter
        float delayedSample = 0.0f;
        size_t bufferSize = plugin->inputBuffer.size();
        
        // Use a fraction of buffer size for delay, minimum 512 samples for audibility
        size_t delayLength = std::max(static_cast<size_t>(512), bufferSize / 4);
        
        // Wait until we have enough samples in buffer
        if (plugin->bufferPos >= delayLength) {
            size_t delayPos = (plugin->bufferPos + bufferSize - delayLength) % bufferSize;
            delayedSample = plugin->inputBuffer[delayPos];
        } else if (bufferSize > delayLength) {
            // Handle initial case when buffer isn't full yet
            size_t delayPos = (plugin->bufferPos + bufferSize - delayLength) % bufferSize;
            delayedSample = plugin->inputBuffer[delayPos];
        }
        
        // Apply pattern alternation system
        float learningRate = plugin->learningRate ? *plugin->learningRate : 0.1f;
        float wfcSample = delayedSample;
        
        // Apply WFC/delay even with basic parameters
        if (bufferSize > 512 && plugin->bufferPos > 512) {
            size_t maxRepeats = plugin->patternRepeats ? static_cast<size_t>(*plugin->patternRepeats) : 2;
            
            // Use detected period for pattern length, fallback to time-based
            size_t patternLength;
            if (plugin->detectedPeriod > 0 && plugin->detectedPeriod < 10000) {
                // Use detected period as pattern length
                patternLength = plugin->detectedPeriod;
            } else {
                // Fallback to 50ms patterns
                patternLength = std::min(static_cast<size_t>(plugin->sampleRate * 0.05f), static_cast<size_t>(2048));
            }
            
            // Initialize pattern system if needed
            if (plugin->currentPattern.empty() && patternLength > 0 && patternLength < 10000) {
                generatePredictedPattern(plugin, plugin->currentPattern, patternLength);
                plugin->patternPlaybackPos = 0;
                plugin->currentRepeatCount = 0;
            }
            
            // Check if we need to switch to next pattern
            if (plugin->patternPlaybackPos >= plugin->currentPattern.size()) {
                plugin->currentRepeatCount++;
                plugin->patternPlaybackPos = 0;
                
                // If we've repeated enough times, switch to next pattern
                if (plugin->currentRepeatCount >= maxRepeats) {
                    if (plugin->backgroundPredictionReady && !plugin->nextPattern.empty()) {
                        // Switch to next pattern
                        plugin->currentPattern = std::move(plugin->nextPattern);
                        plugin->nextPattern.clear();
                        plugin->backgroundPredictionReady = false;
                    } else {
                        // Generate new pattern immediately if background isn't ready
                        generatePredictedPattern(plugin, plugin->currentPattern, patternLength);
                    }
                    plugin->currentRepeatCount = 0;
                    plugin->patternPlaybackPos = 0;
                }
            }
            
            // Generate next pattern in background - simplified timing to avoid crashes
            static uint32_t backgroundCounter = 0;
            if (++backgroundCounter >= 4096 && !plugin->backgroundPredictionReady) { // Every ~80ms
                backgroundCounter = 0;
                if (patternLength > 0 && patternLength < 10000) { // Safety check
                    generatePredictedPattern(plugin, plugin->nextPattern, patternLength);
                    plugin->backgroundPredictionReady = true;
                }
            }
            
            // Get sample from current pattern
            if (!plugin->currentPattern.empty() && plugin->patternPlaybackPos < plugin->currentPattern.size()) {
                float patternSample = plugin->currentPattern[plugin->patternPlaybackPos];
                plugin->patternPlaybackPos++;
                
                // Clamp pattern sample to prevent overload
                patternSample = std::clamp(patternSample, -1.0f, 1.0f);
                
                // Blend pattern with delayed sample
                wfcSample = delayedSample * (1.0f - learningRate) + patternSample * learningRate;
            }
        }
        
        // Mix with original - ensure it always works
        float mixAmount = std::clamp(*plugin->mix, 0.0f, 1.0f);
        
        // If no WFC processing happened, at least pass through the delay
        if (wfcSample == 0.0f && delayedSample != 0.0f) {
            wfcSample = delayedSample;
        }
        
        float outputSample = inputSample * (1.0f - mixAmount) + wfcSample * mixAmount;
        
        // Hard limit output to prevent overload
        plugin->output[i] = std::clamp(outputSample, -1.0f, 1.0f);
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