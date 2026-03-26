#ifndef THERMAL_IMAGE_PROCESSOR_H
#define THERMAL_IMAGE_PROCESSOR_H

#include <stdint.h>
#include <stddef.h>
#include "config.h"
#include "serial_comms.h" // Include to access global configurable parameters

/**
 * @struct DetectedPerson
 * @brief Represents a detected person in the thermal image
 */
typedef struct {
    uint8_t x;              // Centroid X coordinate (0-31)
    uint8_t y;              // Centroid Y coordinate (0-23)
    uint16_t area;          // Number of pixels in detection region
    uint8_t max_distance;   // Maximum distance from centroid (from distance transform)
} DetectedPerson;

/**
 * @struct ThermalProcessor
 * @brief Main image processor state and buffers
 */
typedef struct {
    // Background frame for subtraction (running average)
    uint8_t background[IMAGE_HEIGHT][IMAGE_WIDTH];
    
    // Working buffer for intermediate results (e.g., for morphological operations)
    uint8_t work_buffer[IMAGE_HEIGHT][IMAGE_WIDTH];
    
    // Labeled image for watershed output
    uint8_t labeled_image[IMAGE_HEIGHT][IMAGE_WIDTH];
    
    // Distance transform output
    uint8_t distance_map[IMAGE_HEIGHT][IMAGE_WIDTH];
    
    // Update counter for background frame (not directly used by this struct anymore,
    // bg_init_counter in node.ino manages the background initialization phase)
    // uint16_t background_update_counter; 
    
} ThermalProcessor;

/**
 * @brief Initialize the thermal image processor.
 * Clears all internal buffers to zero.
 * @param processor Processor state structure
 */
void thermal_processor_init(ThermalProcessor* processor);

/**
 * @brief Convert float thermal data to 8-bit grayscale image.
 * Maps temperature range (TEMP_MIN to TEMP_MAX) to 0-255 for image processing.
 * @param thermal_frame Input thermal data (IMAGE_SIZE floats)
 * @param image Output 8-bit grayscale image (IMAGE_HEIGHT x IMAGE_WIDTH)   
 */
void convert_to_8bit_image(const float* thermal_frame, uint8_t image[IMAGE_HEIGHT][IMAGE_WIDTH]);

/**
 * @brief Update the running background average.
 * Uses an exponential moving average.
 * @param processor Processor state
 * @param current_frame Current thermal image (IMAGE_HEIGHT x IMAGE_WIDTH)
 * @param alpha Smoothing factor (0-255, where 255 = full weight to new frame, 0 = no update)
 */
void update_background(ThermalProcessor* processor, const uint8_t current_frame[IMAGE_HEIGHT][IMAGE_WIDTH], uint8_t alpha);

/**
 * @brief Subtract background frame from current frame.
 * @param processor Processor state (uses its stored background frame)
 * @param current_frame Current thermal image
 * @param output Output difference image (IMAGE_HEIGHT x IMAGE_WIDTH)
 */
void subtract_frames(const ThermalProcessor* processor, const uint8_t current_frame[IMAGE_HEIGHT][IMAGE_WIDTH], uint8_t output[IMAGE_HEIGHT][IMAGE_WIDTH]);

/**
 * @brief Apply 3x3 greyscale morphological erosion.
 * @param input Input greyscale image (IMAGE_HEIGHT x IMAGE_WIDTH)
 * @param output Output eroded image (IMAGE_HEIGHT x IMAGE_WIDTH)
 */
void erode_3x3(const uint8_t input[IMAGE_HEIGHT][IMAGE_WIDTH], uint8_t output[IMAGE_HEIGHT][IMAGE_WIDTH]);

/**
 * @brief Apply 3x3 greyscale morphological dilation.
 * @param input Input greyscale image (IMAGE_HEIGHT x IMAGE_WIDTH)
 * @param output Output dilated image (IMAGE_HEIGHT x IMAGE_WIDTH)
 */
void dilate_3x3(const uint8_t input[IMAGE_HEIGHT][IMAGE_WIDTH], uint8_t output[IMAGE_HEIGHT][IMAGE_WIDTH]);

/**
 * @brief Apply 3x3 Gaussian blur.
 * Uses a fixed-point approximation for speed.
 * @param input Input greyscale image (IMAGE_HEIGHT x IMAGE_WIDTH)
 * @param output Output blurred image (IMAGE_HEIGHT x IMAGE_WIDTH)
 */
void gaussian_blur_3x3(const uint8_t input[IMAGE_HEIGHT][IMAGE_WIDTH], uint8_t output[IMAGE_HEIGHT][IMAGE_WIDTH]);

/**
 * @brief Compute Euclidean distance transform.
 * Non-zero pixels in the input are considered "object" and zero pixels "background".
 * Output values represent distance from nearest background pixel.
 * Uses ::DT_BG_THRESHOLD for initial binary conversion.
 * @param input Binary/greyscale image (IMAGE_HEIGHT x IMAGE_WIDTH)
 * @param output Distance map where values represent distance from nearest background pixel
 */
void distance_transform(const uint8_t input[IMAGE_HEIGHT][IMAGE_WIDTH], uint8_t output[IMAGE_HEIGHT][IMAGE_WIDTH]);

/**
 * @brief Watershed algorithm for object segmentation.
 * Identifies distinct objects based on the distance map.
 * Uses ::DT_MAX_DISTANCE, ::MIN_PERSON_AREA, ::MAX_PERSON_AREA for filtering.
 * @param distance_map Distance transform output
 * @param processor Processor state (outputs labeled_image)
 * @param detected_people Output array of detected people
 * @param max_people Maximum number of people to detect
 * @return Number of detected people
 */
uint8_t watershed(const uint8_t distance_map[IMAGE_HEIGHT][IMAGE_WIDTH], 
                  ThermalProcessor* processor,
                  DetectedPerson detected_people[],
                  uint8_t max_people);

/**
 * @brief Complete image processing pipeline.
 * Performs background subtraction, morphological operations, blurring,
 * distance transform, and watershed segmentation to detect people.
 * @param processor Processor state
 * @param current_frame Current thermal image
 * @param detected_people Output array of detected people
 * @param max_people Maximum number of people to detect
 * @return Number of detected people
 */
uint8_t process_thermal_frame(ThermalProcessor* processor,
                              const uint8_t current_frame[IMAGE_HEIGHT][IMAGE_WIDTH],
                              DetectedPerson detected_people[],
                              uint8_t max_people);

#endif // THERMAL_IMAGE_PROCESSOR_H