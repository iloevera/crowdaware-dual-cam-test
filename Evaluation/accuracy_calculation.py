import pandas as pd
import ast
import numpy as np
from scipy.spatial.distance import cdist

# --- Adjustable Constants ---

# CSV file path
CSV_FILE_PATH = "detections.csv"

# Camera Resolutions:
THERMAL_RAW_WIDTH = 32
THERMAL_RAW_HEIGHT = 24
YOLO_WIDTH = 640
YOLO_HEIGHT = 480

# Thermal FOV Adjustment (Cropping Factor):
# The thermal camera's FOV is wider than YOLO's.
THERMAL_CROP_FACTOR = 0.8 

# Minimum Area Thresholds:
# Detections smaller than these thresholds are ignored for accuracy calculations.
YOLO_MIN_AREA_THRESHOLD = 500  # Minimum area (pixels) for a YOLO detection to be considered
THERMAL_MIN_AREA_THRESHOLD = 10 # Minimum area (pixels) for a Thermal detection to be considered

# Matching Distance Threshold:
# This is the maximum Euclidean distance (in the 640x480 YOLO coordinate system)
# between the centroids of a thermal detection and a YOLO detection for them to
# be considered a "match" (True Positive). Adjust based on acceptable positional error.
MATCHING_DISTANCE_THRESHOLD = 20 # pixels

# Area bins for correlation analysis:
# These define ranges for object sizes (areas) to categorize False Negatives (FNs)
# and False Positives (FPs). This helps understand how detection performance
# correlates with object size (a proxy for distance).
# YOLO_AREA_BINS: For YOLO detections (on the 640x480 grid).
# THERMAL_AREA_BINS: For Thermal detections (on the 32x24 grid).
YOLO_AREA_BINS = [0, 2000, 5000, 15000, 30000, np.inf] # Example bins for YOLO area (pixels)
YOLO_AREA_BIN_LABELS = ["<2k", "2k-5k", "5k-15k", "15k-30k", ">30k"]

THERMAL_AREA_BINS = [0, 10, 20, 50, 100, np.inf] # Example bins for Thermal area (pixels)
THERMAL_AREA_BIN_LABELS = ["<10", "10-20", "20-50", "50-100", ">100"]


def calculate_accuracy(csv_file):
    """
    Calculates the accuracy of a thermal camera by comparing its detections
    against a YOLO camera (considered the reference). It performs FOV alignment,
    size-based filtering, object matching, and analyzes False Positives/Negatives
    in relation to object size.
    """
    df = pd.read_csv(csv_file)
    
    # Convert CSV text representations of lists/dictionaries back to actual Python objects
    df['thermal_data'] = df['thermal_data'].apply(ast.literal_eval)
    df['yolo_data'] = df['yolo_data'].apply(ast.literal_eval)

    # Lists to store data for overall metrics and correlation analysis
    total_matched_distances = []
    all_fn_yolo_areas = []
    all_fp_thermal_areas = []
    
    # Counters for total detections considered after filtering
    total_yolo_detections_considered = 0
    total_thermal_detections_considered = 0

    # Pre-calculate Thermal FOV cropping and scaling parameters
    cropped_thermal_width = THERMAL_RAW_WIDTH * THERMAL_CROP_FACTOR
    cropped_thermal_height = THERMAL_RAW_HEIGHT * THERMAL_CROP_FACTOR
    
    offset_x = (THERMAL_RAW_WIDTH - cropped_thermal_width) / 2
    offset_y = (THERMAL_RAW_HEIGHT - cropped_thermal_height) / 2

    scale_x_thermal_to_yolo = YOLO_WIDTH / cropped_thermal_width
    scale_y_thermal_to_yolo = YOLO_HEIGHT / cropped_thermal_height

    # Iterate through each timestamp (row) in the DataFrame
    for index, row in df.iterrows():
        # --- 1. Process Thermal Detections: Filter, Crop, and Scale ---
        current_thermal_coords_scaled = []    # Centroids of thermal detections, scaled to YOLO grid
        current_thermal_original_areas = []   # Original areas of thermal detections (for FP analysis)
        
        for t_det in row['thermal_data']:
            # Apply minimum area threshold to filter out very small thermal detections
            if t_det['area'] < THERMAL_MIN_AREA_THRESHOLD:
                continue

            # Get the thermal detection's centroid coordinates
            thermal_x, thermal_y = t_det['x'], t_det['y']
            
            # Ignore detections that fall outside the defined cropped FOV region
            if not (offset_x <= thermal_x < offset_x + cropped_thermal_width and
                    offset_y <= thermal_y < offset_y + cropped_thermal_height):
                continue

            # Adjust coordinates to be relative to the top-left of the cropped region
            adj_x = thermal_x - offset_x
            adj_y = thermal_y - offset_y

            # Scale the adjusted thermal coordinates to match the YOLO camera's resolution
            scaled_x = adj_x * scale_x_thermal_to_yolo
            scaled_y = adj_y * scale_y_thermal_to_yolo
            
            current_thermal_coords_scaled.append([scaled_x, scaled_y])
            current_thermal_original_areas.append(t_det['area'])
        
        total_thermal_detections_considered += len(current_thermal_coords_scaled)

        # --- 2. Process YOLO Detections: Filter and Calculate Centroids ---
        current_yolo_coords = []          # Centroids of YOLO detections (already in YOLO grid)
        current_yolo_original_areas = []  # Original areas of YOLO detections (for FN analysis)

        for y_det in row['yolo_data']:
            # Extract bounding box coordinates from YOLO detection
            yolo_x1, yolo_y1, yolo_x2, yolo_y2 = y_det['x1'], y_det['y1'], y_det['x2'], y_det['y2']
            
            # Calculate the area of the YOLO bounding box
            yolo_area = (yolo_x2 - yolo_x1) * (yolo_y2 - yolo_y1)
            
            # Apply minimum area threshold to filter out very small YOLO detections
            if yolo_area < YOLO_MIN_AREA_THRESHOLD:
                continue

            # Calculate the centroid of the YOLO bounding box
            cx = (yolo_x1 + yolo_x2) / 2
            cy = (yolo_y1 + yolo_y2) / 2
            
            current_yolo_coords.append([cx, cy])
            current_yolo_original_areas.append(yolo_area)
            
        total_yolo_detections_considered += len(current_yolo_coords)

        # --- 3. Perform Matching between Filtered Thermal and YOLO Detections ---
        # This section tries to find corresponding pairs between the two camera's detections.
        num_thermal = len(current_thermal_coords_scaled)
        num_yolo = len(current_yolo_coords)

        matched_yolo_indices = set()    # Stores indices of YOLO detections that found a match
        matched_thermal_indices = set() # Stores indices of Thermal detections that found a match

        if num_yolo > 0 and num_thermal > 0:
            # Calculate Euclidean distances between every YOLO centroid and every thermal centroid
            distances = cdist(current_yolo_coords, current_thermal_coords_scaled, metric='euclidean')

            # Create a list of all potential matches that are within the MATCHING_DISTANCE_THRESHOLD.
            # Each entry is (distance, yolo_index, thermal_index).
            all_possible_matches = []
            for y_idx in range(num_yolo):
                for t_idx in range(num_thermal):
                    if distances[y_idx, t_idx] < MATCHING_DISTANCE_THRESHOLD:
                        all_possible_matches.append((distances[y_idx, t_idx], y_idx, t_idx))
            
            # Match the closest pairs first.
            all_possible_matches.sort(key=lambda x: x[0]) 

            # Iterate through the sorted matches and assign them if both detections
            # (YOLO and thermal) haven't been matched yet.
            for dist, y_idx, t_idx in all_possible_matches:
                if y_idx not in matched_yolo_indices and t_idx not in matched_thermal_indices:
                    matched_yolo_indices.add(y_idx)
                    matched_thermal_indices.add(t_idx)
                    total_matched_distances.append(dist) # Store distance for spatial accuracy calculation
        
        # --- 4. Identify False Negatives (FNs) and False Positives (FPs) ---
        # False Negatives: YOLO detections that were not successfully matched by any thermal detection.
        for y_idx in range(num_yolo):
            if y_idx not in matched_yolo_indices:
                all_fn_yolo_areas.append(current_yolo_original_areas[y_idx])

        # False Positives: Thermal detections that were not successfully matched by any YOLO detection.
        for t_idx in range(num_thermal):
            if t_idx not in matched_thermal_indices:
                all_fp_thermal_areas.append(current_thermal_original_areas[t_idx])

    # --- 5. Calculate Overall Accuracy Metrics ---
    
    total_fns = len(all_fn_yolo_areas)
    total_fps = len(all_fp_thermal_areas)
    total_tps = len(total_matched_distances) # Each matched distance represents a True Positive

    # Precision: Of all detections made by the thermal camera, how many were correct?
    # TP / (TP + FP)
    precision = total_tps / (total_tps + total_fps) if (total_tps + total_fps) > 0 else 0
    
    # Recall: Of all actual people (as detected by YOLO), how many did the thermal camera find?
    # TP / (TP + FN)
    recall = total_tps / (total_tps + total_fns) if (total_tps + total_fns) > 0 else 0
    
    # F1-Score: The harmonic mean of Precision and Recall, providing a balanced measure
    # of the model's accuracy.
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    # Average pixel offset for successfully matched detections. This indicates
    # the spatial accuracy of the thermal camera relative to YOLO.
    avg_pixel_offset = np.mean(total_matched_distances) if total_matched_distances else 0

    # --- 6. Display Results ---
    print(f"\n--- Thermal Camera Accuracy Analysis ---")
    print(f"Analysis based on {len(df)} timestamps from '{csv_file}'")
    print(f"YOLO Resolution: {YOLO_WIDTH}x{YOLO_HEIGHT}, Thermal Raw Resolution: {THERMAL_RAW_WIDTH}x{THERMAL_RAW_HEIGHT}")
    print(f"Thermal FOV Cropped to {THERMAL_CROP_FACTOR*100:.0f}% central region.")
    print(f"YOLO Min Area Threshold: {YOLO_MIN_AREA_THRESHOLD} pixels, Thermal Min Area Threshold: {THERMAL_MIN_AREA_THRESHOLD} pixels")
    print(f"Matching Distance Threshold: {MATCHING_DISTANCE_THRESHOLD} pixels (on {YOLO_WIDTH}x{YOLO_HEIGHT} grid)")
    print("-" * 60)

    print(f"Total YOLO Detections Considered (after area filter): {total_yolo_detections_considered}")
    print(f"Total Thermal Detections Considered (after area/FOV filter): {total_thermal_detections_considered}")
    print("-" * 60)

    print(f"1. DETECTION ACCURACY")
    print(f"   - True Positives (TP): {total_tps} (YOLO detected & Thermal matched)")
    print(f"   - False Positives (FP): {total_fps} (Thermal detected, but no YOLO match)")
    print(f"   - False Negatives (FN): {total_fns} (YOLO detected, but no Thermal match)")
    print(f"   - Precision: {precision:.2f}")
    print(f"   - Recall: {recall:.2f}")
    print(f"   - F1-Score: {f1_score:.2f}")
    print("-" * 60)

    print(f"2. SPATIAL ACCURACY")
    print(f"   - Avg. Matched Distance Offset: {avg_pixel_offset:.2f} pixels (for TP matches)")
    print("-" * 60)

    print(f"3. FALSE NEGATIVE ANALYSIS (YOLO detections missed by Thermal, by YOLO object size)")
    if all_fn_yolo_areas:
        # Categorize FN YOLO detections by their area using predefined bins
        fn_yolo_counts_by_bin = pd.cut(all_fn_yolo_areas, bins=YOLO_AREA_BINS, labels=YOLO_AREA_BIN_LABELS, right=False)
        fn_yolo_distribution = fn_yolo_counts_by_bin.value_counts().sort_index()
        print(fn_yolo_distribution)
        print(f"   Total FNs: {total_fns}")
    else:
        print("   No False Negatives recorded.")
    print("-" * 60)

    print(f"4. FALSE POSITIVE ANALYSIS (Thermal detections not matched by YOLO, by Thermal object size)")
    if all_fp_thermal_areas:
        # Categorize FP Thermal detections by their area using predefined bins
        fp_thermal_counts_by_bin = pd.cut(all_fp_thermal_areas, bins=THERMAL_AREA_BINS, labels=THERMAL_AREA_BIN_LABELS, right=False)
        fp_thermal_distribution = fp_thermal_counts_by_bin.value_counts().sort_index()
        print(fp_thermal_distribution)
        print(f"   Total FPs: {total_fps}")
    else:
        print("   No False Positives recorded.")
    print("-" * 60)


if __name__ == "__main__":
    # This block creates a dummy CSV file with your provided data for testing purposes.
    # In a real scenario, you would already have your 'detections.csv' file.
    test_csv_content = """timestamp,thermal_count,yolo_count,thermal_data,yolo_data
1772700974,0,0,[],[]
1772700977,0,0,[],[]
1772700980,0,0,[],[]
1772700983,1,0,"[{'x': 1, 'y': 13, 'area': 20}]",[]
1772700986,1,0,"[{'x': 3, 'y': 13, 'area': 22}]",[]
1772700990,1,0,"[{'x': 3, 'y': 12, 'area': 26}]",[]
1772700993,1,0,"[{'x': 4, 'y': 14, 'area': 25}]",[]
1772700996,0,3,[],"[{'x1': 517, 'y1': 190, 'x2': 639, 'y2': 478}, {'x1': 0, 'y1': 334, 'x2': 19, 'y2': 419}, {'x1': 517, 'y1': 192, 'x2': 639, 'y2': 479}]"
1772700999,1,1,"[{'x': 11, 'y': 14, 'area': 32}]","[{'x1': 278, 'y1': 178, 'x2': 492, 'y2': 479}]"
1772701002,1,1,"[{'x': 17, 'y': 15, 'area': 62}]","[{'x1': 129, 'y1': 152, 'x2': 388, Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned'y2': 478}]"
1772701005,1,0,"[{'x': 26, 'y': 15, 'area': 52}]",[]
1772701008,1,1,"[{'x': 28, 'y': 13, 'area': 78}]","[{'x1': 0, 'y1': 135, 'x2': 254, 'y2': 479}]"
1772701011,1,1,"[{'x': 17, 'y': 14, 'area': 63}]","[{'x1': 211, 'y1': 148, 'x2': 468, 'y2': 479}]"
1772701014,1,1,"[{'x': 16, 'y': 14, 'area': 65}]","[{'x1': 296, 'y1': 131, 'x2': 459, 'y2': 478}]"
1772701017,1,1,"[{'x': 9, 'y': 14, 'area': 50}]","[{'x1': 459, 'y1': 109, 'x2': 627, 'y2': 477}]"
1772701020,1,1,"[{'x': 9, 'y': 14, 'area': 60}]","[{'x1': 415, 'y1': 137, 'x2': 639, 'y2': 477}]"
1772701023,1,1,"[{'x': 10, 'y': 13, 'area': 60}]","[{'x1': 416, 'y1': 125, 'x2': 638, 'y2': 478}]"
1772701026,1,1,"[{'x': 10, 'y': 14, 'area': 68}]","[{'x1': 342, 'y1': 167, 'x2': 580, 'y2': 478}]"
1772701029,1,1,"[{'x': 11, 'y': 15, 'area': 47}]","[{'x1': 381, 'y1': 163, 'x2': 545, 'y2': 477}]"
1772701032,1,1,"[{'x': 13, 'y': 15, 'area': 45}]","[{'x1': 271, 'y1': 167, 'x2': 417, 'y2': 478}]"
1772701035,1,1,"[{'x': 16, 'y': 14, 'area': 32}]","[{'x1': 234, 'y1': 162, 'x2': 389, 'y2': 479}]"
1772701038,1,1,"[{'x': 16, 'y': 12, 'area': 45}]","[{'x1': 265, 'y1': 151, 'x2': 414, 'y2': 476}]"
1772701042,1,1,"[{'x': 16, 'y': 12, 'area': 57}]","[{'x1': 250, 'y1': 93, 'x2': 425, 'y2': 477}]"
1772701045,0,2,[],"[{'x1': 249, 'y1': 166, 'x2': 391, 'y2': 476}, {'x1': 249, 'y1': 163, 'x2': 392, 'y2': 475}]"
1772701048,0,1,[],"[{'x1': 241, 'y1': 189, 'x2': 413, 'y2': 476}]"
1772701051,0,1,[],"[{'x1': 196, 'y1': 174, 'x2': 372, 'y2': 476}]"
1772701054,0,1,[],"[{'x1': 45, 'y1': 192, 'x2': 235, 'y2': 479}]"
1772701057,0,1,[],"[{'x1': 247, 'y1': 190, 'x2': 385, 'y2': 478}]"
1772701060,0,3,[],"[{'x1': 317, 'y1': 134, 'x2': 560, 'y2': 478}, {'x1': 11, 'y1': 183, 'x2': 234, 'y2': 478}, {'x1': 0, 'y1': 281, 'x2': 50, 'y2': 416}]"
1772701063,1,3,"[{'x': 12, 'y': 11, 'area': 27}]","[{'x1': 331, 'y1': 145, 'x2': 562, 'y2': 477}, {'x1': 69, 'y1': 268, 'x2': 152, 'y2': 474}, {'x1': 0, 'y1': 253, 'x2': 68, 'y2': 477}]"
1772701066,1,3,"[{'x': 12, 'y': 12, 'area': 26}]","[{'x1': 331, 'y1': 149, 'x2': 560, 'y2': 477}, {'x1': 0, 'y1': 217, 'x2': 150, 'y2': 477}, {'x1': 0, 'y1': 270, 'x2': 27, 'y2': 339}]"
1772701069,1,1,"[{'x': 13, 'y': 12, 'area': 35}]","[{'x1': 307, 'y1': 157, 'x2': 554, 'y2': 477}]"
1772701072,1,1,"[{'x': 13, 'y': 11, 'area': 38}]","[{'x1': 292, 'y1': 149, 'x2': 526, 'y2': 477}]"
1772701075,1,2,"[{'x': 13, 'y': 12, 'area': 31}]","[{'x1': 349, 'y1': 175, 'x2': 500, 'y2': 476}, {'x1': 349, 'y1': 175, 'x2': 498, 'y2': 477}]"
1772701078,0,1,[],"[{'x1': 569, 'y1': 195, 'x2': 639, 'y2': 471}]"
1772701081,0,0,[],[]
1772701084,0,1,[],"[{'x1': 166, 'y1': 183, 'x2': 359, 'y2': 478}]"
1772701087,1,0,"[{'x': 25, 'y': 9, 'area': 22}]",[]
1772701090,0,1,[],"[{'x1': 0, 'y1': 334, 'x2': 20, 'y2': 480}]"
1772701094,0,2,[],"[{'x1': 145, 'y1': 173, 'x2': 301, 'y2': 477}, {'x1': 20, 'y1': 294, 'x2': 56, 'y2': 325}]"
1772701097,0,0,[],[]
"""
    with open('detections.csv', 'w', newline='') as f:
        f.write(test_csv_content)

    calculate_accuracy('detections.csv')