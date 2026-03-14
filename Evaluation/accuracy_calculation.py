import pandas as pd
import ast
import numpy as np
from scipy.spatial.distance import cdist

def calculate_accuracy(csv_file):
    # Load the data
    df = pd.read_csv(csv_file)
    
    df['thermal_data'] = df['thermal_data'].apply(ast.literal_eval)
    df['yolo_data'] = df['yolo_data'].apply(ast.literal_eval)

    # Difference between the number of people detected (lower >> better)
    df['count_diff'] = abs(df['thermal_count'] - df['yolo_count'])
    mae_count = df['count_diff'].mean()
    
    # Calculate how often the counts matched exactly (percentage)
    perfect_match_rate = (df['thermal_count'] == df['yolo_count']).mean() * 100

    # Difference of the location detected 
    all_distances = []
    FP = 0 #False Positive
    FN = 0 #False Negative

    for index, row in df.iterrows():
        timestamp = row['timestamp']

        thermal_pts = []
        yolo_pts = []
        
        # Extract Thermal Centroids (already x, y)
        for t in row['thermal_data']:
            thermal_pts.append([t['x'], t['y']])
            
        # Extract YOLO Centroids (converting x1, y1, x2, y2 to center x, y)
        # Note: We scale by 0.5 because RGB (640 * 480) is 2x Thermal display (320 * 240)
        for y in row['yolo_data']:
            cx = ((y['x1'] + y['x2']) / 2) * 0.5 
            cy = ((y['y1'] + y['y2']) / 2) * 0.5
            yolo_pts.append([cx, cy])
            
        # If both cameras detected people, find the closest pairs
        if len(yolo_pts) > 0:
            if len(thermal_pts) > 0:
                # Calculate distance from EVERY YOLO point to EVERY Thermal point
                distances = cdist(yolo_pts, thermal_pts, metric='euclidean')
                
                # For EACH YOLO point, find the closest Thermal point
                # If YOLO = 2 and Thermal = 1, both YOLO points will calculate 
                # their distance to the single Thermal point.
                min_distances = np.min(distances, axis=1)
                all_distances.extend(min_distances)
                
            else:
                # YOLO detected people, but Thermal detected 0 people at this timestamp
                FN += len(yolo_pts)
        else:
            if len(thermal_pts) > 0:
                # YOLO detected 0 people, but Thermal detected someone (False Positive)
                FP += len(thermal_pts)

    # Calculate average pixel offset for the points that were matched
    avg_pixel_offset = np.mean(all_distances) if all_distances else 0

    # Display Results\
    print(f"\n\nFrom {len(df)} timestamps in {csv_file} ")
    print(f"1. COUNT ACCURACY")
    print(f"   - Mean Absolute Error: {mae_count:.2f} people")
    print(f"   - Thermal Ghost Detections(FP) : {FP}")
    print(f"   - Thermal Missed Detection(FN) : {FN}")
    print("-" * 33)
    print(f"2. SPATIAL ACCURACY")
    print(f"   - Avg. Distance Offset: {avg_pixel_offset:.2f} pixels\n\n")


if __name__ == "__main__":
    calculate_accuracy('detections.csv')
