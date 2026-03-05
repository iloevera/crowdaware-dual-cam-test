#include <Wire.h>
#include "mlx_sensor.h"
#include "thermal_image_processor.h"
#include "config.h"

// MLX sensor instance
MlxSensor mlx_sensor;

// Thermal processor instance
ThermalProcessor thermal_processor;

// Frame buffer for raw MLX90640 data (768 pixels)
float mlx_frame[768];

// Converted 8-bit thermal image (24x32)
uint8_t thermal_image[IMAGE_HEIGHT][IMAGE_WIDTH];

// Detected people array
DetectedPerson detected_people[MAX_PEOPLE];

// Background initialization counter
uint16_t bg_init_counter = 0;

void setup() {
    Serial.begin(57600);
    while (!Serial);
    
    DEBUG_PRINTLN("=== CrowdAware Node 2026 ===");
    DEBUG_PRINTLN("Initializing MLX90640 thermal sensor...");
    
    // Initialize MLX sensor with custom I2C pins (adjust as needed)
    // For Arduino boards: SDA=41, SCL=42, Speed=1MHz
    if (!mlx_sensor.init(MLX90640_ADDRESS, 41, 42, MLX_I2C_SPEED)) {
        DEBUG_PRINTLN("ERROR: Failed to initialize MLX90640!");
        while (1);
    }
    
    DEBUG_PRINTLN("MLX90640 initialized successfully.");
    
    // Initialize thermal processor
    thermal_processor_init(&thermal_processor);
    DEBUG_PRINTLN("Thermal processor initialized.");
    
    DEBUG_PRINTLN("Beginning background frame initialization...");
    DEBUG_PRINT("Frames remaining: ");
    DEBUG_PRINTLN(BG_FRAME_COUNT);
}

void loop() {
    // Read thermal frame from MLX90640
    if (!mlx_sensor.readFrame(mlx_frame)) {
        delay(10);
        return;
    }
    
    // Convert float values to 8-bit grayscale (0-255)
    convert_to_8bit_image(mlx_frame, thermal_image);

    // Background initialization phase
    if (bg_init_counter < BG_FRAME_COUNT) {
        // Use exponential moving average for background (lower alpha for slower convergence)
        uint8_t alpha = 25; // Slow adaptation (0-255 scale)
        update_background(&thermal_processor, (const uint8_t (*)[IMAGE_WIDTH])thermal_image, alpha);
        
        bg_init_counter++;
        
        if (bg_init_counter % 5 == 0) {
            DEBUG_PRINT("Background init: ");
            DEBUG_PRINT(BG_FRAME_COUNT - bg_init_counter);
            DEBUG_PRINTLN(" frames remaining");
        }
        return; // Skip processing during background initialization
    }

    // Process thermal frame to detect people
    uint8_t num_detected = process_thermal_frame(
        &thermal_processor,
        (const uint8_t (*)[IMAGE_WIDTH])thermal_image,
        detected_people,
        MAX_PEOPLE
    );
    
    if (num_detected == 0) {
        // Update background only when no people are detected to avoid contamination
        uint8_t bg_alpha = 5; // Slower background update after initialization
        update_background(&thermal_processor, (const uint8_t (*)[IMAGE_WIDTH])thermal_image, bg_alpha);
    }
    
    // Output results
    output_detection_results(num_detected, &thermal_processor);
}

/**
 * @brief Output detected human coordinates to serial
 * Format depends on SERIAL_OUTPUT_MODE in config.h
 */
void output_detection_results(uint8_t num_detected, ThermalProcessor* processor) {
#if SERIAL_OUTPUT_MODE == 1
    // Packet structure:
    // orig(768) + step1(768) + step2(768) + step3(768) +
    // num_detected(1) + 4*num_detected

    uint8_t header[4] = {0xFE, 0x01, 0xFE, 0x01};
    Serial.write((uint8_t*)header, 4);
    uint16_t packet_size = IMAGE_SIZE * 4 + 1 + (4 * num_detected);
    Serial.write((uint8_t*)&packet_size, sizeof(packet_size));

    // send four images in row-major order
    Serial.write((uint8_t*)thermal_image, IMAGE_SIZE);
    Serial.write((uint8_t*)processor->background, IMAGE_SIZE);
    Serial.write((uint8_t*)processor->distance_map, IMAGE_SIZE);
    Serial.write((uint8_t*)processor->labeled_image, IMAGE_SIZE);

    Serial.write(num_detected);
    for (uint8_t i = 0; i < num_detected; i++) {
        Serial.write(detected_people[i].y);
        Serial.write(detected_people[i].x);
        Serial.write((uint8_t*)&detected_people[i].area, sizeof(uint16_t));
    }
#else
    // Human-readable format (original)
    DEBUG_PRINT("Detected: ");
    DEBUG_PRINT(num_detected);
    DEBUG_PRINTLN(" person(s)");
    
    if (num_detected > 0) {
        for (uint8_t i = 0; i < num_detected; i++) {
            DEBUG_PRINT("  Person ");
            DEBUG_PRINT(i + 1);
            DEBUG_PRINT(": X=");
            DEBUG_PRINT(detected_people[i].x);
            DEBUG_PRINT(" Y=");
            DEBUG_PRINT(detected_people[i].y);
            DEBUG_PRINT(" Area=");
            DEBUG_PRINT(detected_people[i].area);
            DEBUG_PRINT(" pixels, MaxDist=");
            DEBUG_PRINTLN(detected_people[i].max_distance);
        }
    }
    
    DEBUG_PRINTLN();
#endif
}