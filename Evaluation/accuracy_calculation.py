import pandas as pd
import ast
import numpy as np
import math
from scipy.spatial.distance import cdist
import tkinter as tk
from tkinter import filedialog, scrolledtext, ttk, messagebox
from PIL import Image, ImageTk 
import os

# --- 3D Overhead Projection Constants ---
CEILING_HEIGHT_CM = 450
PERSON_HEIGHT_CM = 170
DISTANCE_Z_CM = CEILING_HEIGHT_CM - PERSON_HEIGHT_CM

THERMAL_RAW_WIDTH = 32
THERMAL_RAW_HEIGHT = 24
YOLO_WIDTH = 640
YOLO_HEIGHT = 480

# FOVs in radians
RGB_FOV_H = math.radians(102)
RGB_FOV_V = math.radians(67)
THERM_FOV_H = math.radians(110)
THERM_FOV_V = math.radians(75)

# Focal lengths
RGB_FX = (YOLO_WIDTH / 2) / math.tan(RGB_FOV_H / 2)
RGB_FY = (YOLO_HEIGHT / 2) / math.tan(RGB_FOV_V / 2)
THERM_FX = (THERMAL_RAW_WIDTH / 2) / math.tan(THERM_FOV_H / 2)
THERM_FY = (THERMAL_RAW_HEIGHT / 2) / math.tan(THERM_FOV_V / 2)

# Physical Offset (Thermal relative to Pi Cam)
OFFSET_X_CM = -1.7
OFFSET_Y_CM = -0.7

# --- Projection Functions ---
def yolo_to_thermal(yolo_x, yolo_y):
    """Maps a YOLO pixel coordinate to the Thermal pixel space"""
    real_x_cm = ((yolo_x - (YOLO_WIDTH / 2)) * DISTANCE_Z_CM) / RGB_FX
    real_y_cm = ((yolo_y - (YOLO_HEIGHT / 2)) * DISTANCE_Z_CM) / RGB_FY
    
    therm_real_x_cm = real_x_cm + OFFSET_X_CM
    therm_real_y_cm = real_y_cm + OFFSET_Y_CM
    
    therm_x = ((therm_real_x_cm * THERM_FX) / DISTANCE_Z_CM) + (THERMAL_RAW_WIDTH / 2)
    therm_y = ((therm_real_y_cm * THERM_FY) / DISTANCE_Z_CM) + (THERMAL_RAW_HEIGHT / 2)
    
    return therm_x, therm_y

def thermal_to_yolo(therm_x, therm_y):
    """Maps a Thermal pixel coordinate to the YOLO pixel space"""
    therm_real_x_cm = ((therm_x - (THERMAL_RAW_WIDTH / 2)) * DISTANCE_Z_CM) / THERM_FX
    therm_real_y_cm = ((therm_y - (THERMAL_RAW_HEIGHT / 2)) * DISTANCE_Z_CM) / THERM_FY
    
    real_x_cm = therm_real_x_cm - OFFSET_X_CM
    real_y_cm = therm_real_y_cm - OFFSET_Y_CM
    
    yolo_x = ((real_x_cm * RGB_FX) / DISTANCE_Z_CM) + (YOLO_WIDTH / 2)
    yolo_y = ((real_y_cm * RGB_FY) / DISTANCE_Z_CM) + (YOLO_HEIGHT / 2)
    
    return yolo_x, yolo_y


# --- Adjustable Constants (Default Values) ---
YOLO_MIN_AREA_THRESHOLD_DEFAULT = 10000
THERMAL_MIN_AREA_THRESHOLD_DEFAULT = 10
MATCHING_DISTANCE_THRESHOLD_DEFAULT = 150 # Adjusted lower since it maps accurately now

# Area bins for correlation analysis:
YOLO_AREA_BINS_DEFAULT = [0, 2000, 5000, 15000, 30000, np.inf]
YOLO_AREA_BIN_LABELS_DEFAULT = ["<2k", "2k-5k", "5k-15k", "15k-30k", ">30k"]

THERMAL_AREA_BINS_DEFAULT = [0, 10, 20, 50, 100, np.inf]
THERMAL_AREA_BIN_LABELS_DEFAULT = ["<10", "10-20", "20-50", "50-100", ">100"]

class AccuracyAnalyzerGUI:
    def __init__(self, master):
        self.master = master
        master.title("Thermal Camera Accuracy Analyzer (3D Parallax Corrected)")
        master.geometry("1400x800")

        # --- Variables to hold GUI state ---
        self.csv_file_path = tk.StringVar(value="")
        self.image_dir_path = tk.StringVar(value="")

        self.yolo_min_area_threshold = tk.IntVar(value=YOLO_MIN_AREA_THRESHOLD_DEFAULT)
        self.thermal_min_area_threshold = tk.IntVar(value=THERMAL_MIN_AREA_THRESHOLD_DEFAULT)
        self.matching_distance_threshold = tk.IntVar(value=MATCHING_DISTANCE_THRESHOLD_DEFAULT)

        self.yolo_area_bins_str = tk.StringVar(value=", ".join(map(str, YOLO_AREA_BINS_DEFAULT[:-1])) + ", inf")
        self.yolo_area_bin_labels_str = tk.StringVar(value=", ".join(YOLO_AREA_BIN_LABELS_DEFAULT))
        self.thermal_area_bins_str = tk.StringVar(value=", ".join(map(str, THERMAL_AREA_BINS_DEFAULT[:-1])) + ", inf")
        self.thermal_area_bin_labels_str = tk.StringVar(value=", ".join(THERMAL_AREA_BIN_LABELS_DEFAULT))

        self.analysis_results = {}
        self.image_events = [] 
        self.current_image_index = 0
        self.filtered_image_events = []
        self.image_display_filter = tk.StringVar(value="All") 
        self.current_photo = None 

        self.create_widgets()

    def create_widgets(self):
        param_frame = ttk.LabelFrame(self.master, text="Parameters", padding="10")
        param_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

        results_frame = ttk.LabelFrame(self.master, text="Analysis Results", padding="10")
        results_frame.grid(row=0, column=1, rowspan=2, padx=10, pady=10, sticky="nsew")

        image_review_frame = ttk.LabelFrame(self.master, text="Image Review", padding="10")
        image_review_frame.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")

        self.master.grid_rowconfigure(0, weight=1)
        self.master.grid_rowconfigure(1, weight=1)
        self.master.grid_columnconfigure(0, weight=1)
        self.master.grid_columnconfigure(1, weight=1)

        row_idx = 0
        ttk.Label(param_frame, text="CSV File:").grid(row=row_idx, column=0, sticky="w", pady=2)
        ttk.Entry(param_frame, textvariable=self.csv_file_path, width=40).grid(row=row_idx, column=1, sticky="ew", pady=2)
        ttk.Button(param_frame, text="Browse...", command=self.browse_csv).grid(row=row_idx, column=2, sticky="w", padx=5, pady=2)
        row_idx += 1

        ttk.Label(param_frame, text="Image Directory:").grid(row=row_idx, column=0, sticky="w", pady=2)
        ttk.Entry(param_frame, textvariable=self.image_dir_path, width=40).grid(row=row_idx, column=1, sticky="ew", pady=2)
        ttk.Button(param_frame, text="Browse...", command=self.browse_image_dir).grid(row=row_idx, column=2, sticky="w", padx=5, pady=2)
        row_idx += 1

        ttk.Separator(param_frame, orient="horizontal").grid(row=row_idx, column=0, columnspan=3, sticky="ew", pady=5)
        row_idx += 1

        ttk.Label(param_frame, text="YOLO Min Area Threshold:").grid(row=row_idx, column=0, sticky="w", pady=2)
        ttk.Entry(param_frame, textvariable=self.yolo_min_area_threshold, width=10).grid(row=row_idx, column=1, sticky="w", pady=2)
        row_idx += 1

        ttk.Label(param_frame, text="Thermal Min Area Threshold:").grid(row=row_idx, column=0, sticky="w", pady=2)
        ttk.Entry(param_frame, textvariable=self.thermal_min_area_threshold, width=10).grid(row=row_idx, column=1, sticky="w", pady=2)
        row_idx += 1

        ttk.Label(param_frame, text="Matching Dist Threshold (YOLO pixels):").grid(row=row_idx, column=0, sticky="w", pady=2)
        ttk.Entry(param_frame, textvariable=self.matching_distance_threshold, width=10).grid(row=row_idx, column=1, sticky="w", pady=2)
        row_idx += 1
        
        ttk.Label(param_frame, text="YOLO Area Bins (comma-separated, end with inf):").grid(row=row_idx, column=0, sticky="w", pady=2)
        ttk.Entry(param_frame, textvariable=self.yolo_area_bins_str, width=40).grid(row=row_idx, column=1, columnspan=2, sticky="ew", pady=2)
        row_idx += 1

        ttk.Label(param_frame, text="YOLO Area Bin Labels (comma-separated):").grid(row=row_idx, column=0, sticky="w", pady=2)
        ttk.Entry(param_frame, textvariable=self.yolo_area_bin_labels_str, width=40).grid(row=row_idx, column=1, columnspan=2, sticky="ew", pady=2)
        row_idx += 1

        ttk.Label(param_frame, text="Thermal Area Bins (comma-separated, end with inf):").grid(row=row_idx, column=0, sticky="w", pady=2)
        ttk.Entry(param_frame, textvariable=self.thermal_area_bins_str, width=40).grid(row=row_idx, column=1, columnspan=2, sticky="ew", pady=2)
        row_idx += 1

        ttk.Label(param_frame, text="Thermal Area Bin Labels (comma-separated):").grid(row=row_idx, column=0, sticky="w", pady=2)
        ttk.Entry(param_frame, textvariable=self.thermal_area_bin_labels_str, width=40).grid(row=row_idx, column=1, columnspan=2, sticky="ew", pady=2)
        row_idx += 1

        ttk.Button(param_frame, text="Calculate Accuracy", command=self.run_analysis).grid(row=row_idx, column=0, columnspan=3, pady=10)
        
        param_frame.grid_columnconfigure(1, weight=1) 

        # --- Results Frame Widgets ---
        self.results_text = scrolledtext.ScrolledText(results_frame, wrap="word", width=60, height=20, state="disabled")
        self.results_text.pack(expand=True, fill="both")

        # --- Image Review Frame Widgets ---
        filter_frame = ttk.Frame(image_review_frame)
        filter_frame.pack(pady=5)
        ttk.Label(filter_frame, text="Show:").pack(side="left", padx=5)
        ttk.Radiobutton(filter_frame, text="All", variable=self.image_display_filter, value="All", command=self.apply_image_filter).pack(side="left", padx=5)
        ttk.Radiobutton(filter_frame, text="True Positives", variable=self.image_display_filter, value="TP", command=self.apply_image_filter).pack(side="left", padx=5)
        ttk.Radiobutton(filter_frame, text="False Positives", variable=self.image_display_filter, value="FP", command=self.apply_image_filter).pack(side="left", padx=5)
        ttk.Radiobutton(filter_frame, text="False Negatives", variable=self.image_display_filter, value="FN", command=self.apply_image_filter).pack(side="left", padx=5)

        self.image_label = ttk.Label(image_review_frame, text="No image loaded.")
        self.image_label.pack(pady=10)
        
        self.image_info_label = ttk.Label(image_review_frame, text="Image Info: ")
        self.image_info_label.pack(pady=2)

        nav_frame = ttk.Frame(image_review_frame)
        nav_frame.pack(pady=5)
        ttk.Button(nav_frame, text="Previous", command=self.show_previous_image).pack(side="left", padx=10)
        ttk.Button(nav_frame, text="Next", command=self.show_next_image).pack(side="left", padx=10)

    def browse_csv(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if file_path:
            self.csv_file_path.set(file_path)
            self.image_dir_path.set(os.path.dirname(file_path))

    def browse_image_dir(self):
        dir_path = filedialog.askdirectory()
        if dir_path:
            self.image_dir_path.set(dir_path)

    def parse_bins(self, bins_str_var, labels_str_var, default_bins, default_labels):
        try:
            bins_raw = [x.strip() for x in bins_str_var.get().split(',')]
            bins_list = []
            for x in bins_raw:
                if x.lower() == 'inf':
                    bins_list.append(np.inf)
                else:
                    bins_list.append(float(x))
            
            labels_list = [x.strip() for x in labels_str_var.get().split(',')]
            
            if len(bins_list) != len(labels_list) + 1:
                raise ValueError(f"Number of bin edges ({len(bins_list)}) must be one more than number of labels ({len(labels_list)}).")
            return bins_list, labels_list
        except Exception as e:
            messagebox.showerror("Input Error", f"Invalid bin format: {e}\nUsing default bins for this category.")
            return default_bins, default_labels

    def run_analysis(self):
        csv_path = self.csv_file_path.get()
        if not os.path.exists(csv_path):
            messagebox.showerror("Error", "CSV file not found.")
            return

        yolo_bins, yolo_labels = self.parse_bins(self.yolo_area_bins_str, self.yolo_area_bin_labels_str, YOLO_AREA_BINS_DEFAULT, YOLO_AREA_BIN_LABELS_DEFAULT)
        thermal_bins, thermal_labels = self.parse_bins(self.thermal_area_bins_str, self.thermal_area_bin_labels_str, THERMAL_AREA_BINS_DEFAULT, THERMAL_AREA_BIN_LABELS_DEFAULT)

        try:
            results, image_events = self._calculate_accuracy(
                csv_path,
                self.yolo_min_area_threshold.get(),
                self.thermal_min_area_threshold.get(),
                self.matching_distance_threshold.get(),
                yolo_bins, yolo_labels,
                thermal_bins, thermal_labels
            )
            self.analysis_results = results
            self.image_events = image_events
            self.display_results()
            self.apply_image_filter() 
        except Exception as e:
            messagebox.showerror("Analysis Error", f"An error occurred during analysis: {e}")

    def display_results(self):
        self.results_text.config(state="normal")
        self.results_text.delete(1.0, tk.END)
        
        res = self.analysis_results
        
        output = f"--- Thermal Camera Accuracy Analysis ---\n"
        output += f"Analysis based on {res['num_timestamps']} timestamps from '{self.csv_file_path.get()}'\n"
        output += f"Overhead Configuration: Room {CEILING_HEIGHT_CM}cm | Person {PERSON_HEIGHT_CM}cm\n"
        output += f"3D Parallax Correction Applied (FOV Aligned to Overlap Region)\n"
        output += f"YOLO Min Area Threshold: {self.yolo_min_area_threshold.get()} pixels | Thermal Min Area: {self.thermal_min_area_threshold.get()} pixels\n"
        output += f"Matching Distance Threshold: {self.matching_distance_threshold.get()} pixels (on {YOLO_WIDTH}x{YOLO_HEIGHT} grid)\n"
        output += "-" * 60 + "\n\n"

        output += f"Total YOLO Detections Considered (in FOV overlap): {res['total_yolo_detections_considered']}\n"
        output += f"Total Thermal Detections Considered (in FOV overlap): {res['total_thermal_detections_considered']}\n"
        output += "-" * 60 + "\n\n"

        output += f"1. DETECTION ACCURACY\n"
        output += f"   - True Positives (TP): {res['total_tps']} ({res['tp_percentage']:.2f}% of relevant events)\n"
        output += f"   - False Positives (FP): {res['total_fps']} ({res['fp_percentage']:.2f}% of relevant events)\n"
        output += f"   - False Negatives (FN): {res['total_fns']} ({res['fn_percentage']:.2f}% of relevant events)\n"
        output += f"   - Precision: {res['precision']:.2f}\n"
        output += f"   - Recall: {res['recall']:.2f}\n"
        output += f"   - F1-Score: {res['f1_score']:.2f}\n"
        output += "-" * 60 + "\n\n"

        output += f"2. SPATIAL ACCURACY\n"
        output += f"   - Avg. Matched Distance Offset: {res['avg_pixel_offset']:.2f} pixels (YOLO Scale)\n"
        output += "-" * 60 + "\n\n"

        output += f"3. FALSE NEGATIVE ANALYSIS (YOLO detections missed by Thermal)\n"
        if res['fn_yolo_distribution'] is not None:
            output += str(res['fn_yolo_distribution']) + "\n"
            output += f"   Total FNs: {res['total_fns']}\n"
        else:
            output += "   No False Negatives recorded.\n"
        output += "-" * 60 + "\n\n"

        output += f"4. FALSE POSITIVE ANALYSIS (Thermal detections not matched by YOLO)\n"
        if res['fp_thermal_distribution'] is not None:
            output += str(res['fp_thermal_distribution']) + "\n"
            output += f"   Total FPs: {res['total_fps']}\n"
        else:
            output += "   No False Positives recorded.\n"
        output += "-" * 60 + "\n\n"

        self.results_text.insert(tk.END, output)
        self.results_text.config(state="disabled")

    def apply_image_filter(self):
        filter_type = self.image_display_filter.get()
        if filter_type == "All":
            self.filtered_image_events = self.image_events
        else:
            self.filtered_image_events = [event for event in self.image_events if event['type'] == filter_type]
        
        self.current_image_index = 0
        self.show_current_image()

    def show_current_image(self):
        if not self.filtered_image_events:
            self.image_label.config(image="", text="No images to display for this filter.")
            self.image_info_label.config(text="Image Info: ")
            self.current_photo = None
            return

        image_dir = self.image_dir_path.get()
        if not image_dir or not os.path.exists(image_dir):
            self.image_label.config(image="", text="Image directory not set or invalid.")
            self.image_info_label.config(text="Image Info: ")
            self.current_photo = None
            return

        event = self.filtered_image_events[self.current_image_index]
        timestamp = event['timestamp']
        image_filename = os.path.join(image_dir, f"frame_{timestamp}.jpg")

        if os.path.exists(image_filename):
            try:
                img = Image.open(image_filename)
                img.thumbnail((400, 300))
                self.current_photo = ImageTk.PhotoImage(img)
                self.image_label.config(image=self.current_photo, text="")
                
                info_text = f"Type: {event['type']} | Timestamp: {timestamp}"
                if event.get('yolo_area') is not None:
                    info_text += f" | YOLO Area: {event['yolo_area']:.0f}"
                if event.get('thermal_area') is not None:
                    info_text += f" | Thermal Area: {event['thermal_area']:.0f}"
                if event.get('distance') is not None:
                    info_text += f" | Match Dist: {event['distance']:.2f}"
                info_text += f" ({self.current_image_index + 1} / {len(self.filtered_image_events)})"
                self.image_info_label.config(text=info_text)

            except Exception as e:
                self.image_label.config(image="", text=f"Error loading image: {e}")
                self.image_info_label.config(text=f"Image Info: Error loading {image_filename}")
                self.current_photo = None
        else:
            self.image_label.config(image="", text=f"Image not found: {image_filename}")
            self.image_info_label.config(text=f"Image Info: Missing {image_filename}")
            self.current_photo = None

    def show_next_image(self):
        if self.filtered_image_events:
            self.current_image_index = (self.current_image_index + 1) % len(self.filtered_image_events)
            self.show_current_image()

    def show_previous_image(self):
        if self.filtered_image_events:
            self.current_image_index = (self.current_image_index - 1 + len(self.filtered_image_events)) % len(self.filtered_image_events)
            self.show_current_image()

    def _calculate_accuracy(self, csv_file, yolo_min_area_threshold, thermal_min_area_threshold,
                            matching_distance_threshold, yolo_area_bins, yolo_area_bin_labels,
                            thermal_area_bins, thermal_area_bin_labels):
        
        df = pd.read_csv(csv_file)
        
        df['thermal_data'] = df['thermal_data'].apply(ast.literal_eval)
        df['yolo_data'] = df['yolo_data'].apply(ast.literal_eval)

        total_matched_distances = []
        all_fn_yolo_areas = []
        all_fp_thermal_areas = []
        image_events_for_gui = []

        total_yolo_detections_considered = 0
        total_thermal_detections_considered = 0

        # Boundary tolerance buffer
        THERM_BUFFER = 2.0  
        RGB_BUFFER = 30.0    

        for index, row in df.iterrows():
            timestamp = row['timestamp']
            
            # --- 1. Process Thermal Detections ---
            current_thermal_coords_scaled = []    
            current_thermal_original_areas = []   
            
            for t_det in row['thermal_data']:
                if t_det['area'] < thermal_min_area_threshold:
                    continue

                thermal_x, thermal_y = t_det['x'], t_det['y']
                
                # Project into YOLO space to check bounds and compute distances
                mapped_yolo_x, mapped_yolo_y = thermal_to_yolo(thermal_x, thermal_y)
                
                # Boundary check: Ensure the point is within the RGB camera's view (+ buffer)
                if (-RGB_BUFFER <= mapped_yolo_x <= YOLO_WIDTH + RGB_BUFFER) and \
                   (-RGB_BUFFER <= mapped_yolo_y <= YOLO_HEIGHT + RGB_BUFFER):
                    
                    # Store as YOLO coordinates for distance calculation
                    current_thermal_coords_scaled.append([mapped_yolo_x, mapped_yolo_y])
                    current_thermal_original_areas.append(t_det['area'])
            
            total_thermal_detections_considered += len(current_thermal_coords_scaled)

            # --- 2. Process YOLO Detections ---
            current_yolo_coords = []          
            current_yolo_original_areas = []  

            for y_det in row['yolo_data']:
                yolo_x1, yolo_y1, yolo_x2, yolo_y2 = y_det['x1'], y_det['y1'], y_det['x2'], y_det['y2']
                yolo_area = (yolo_x2 - yolo_x1) * (yolo_y2 - yolo_y1)
                
                if yolo_area < yolo_min_area_threshold:
                    continue

                cx = (yolo_x1 + yolo_x2) / 2
                cy = (yolo_y1 + yolo_y2) / 2
                
                # Project into Thermal space to check bounds
                mapped_therm_x, mapped_therm_y = yolo_to_thermal(cx, cy)
                
                # Boundary check: Ensure point is within Thermal camera's view (+ buffer)
                if (-THERM_BUFFER <= mapped_therm_x <= THERMAL_RAW_WIDTH + THERM_BUFFER) and \
                   (-THERM_BUFFER <= mapped_therm_y <= THERMAL_RAW_HEIGHT + THERM_BUFFER):
                    
                    current_yolo_coords.append([cx, cy])
                    current_yolo_original_areas.append(yolo_area)
                
            total_yolo_detections_considered += len(current_yolo_coords)

            # --- 3. Perform Matching ---
            num_thermal = len(current_thermal_coords_scaled)
            num_yolo = len(current_yolo_coords)

            matched_yolo_indices = set()
            matched_thermal_indices = set()

            if num_yolo > 0 and num_thermal > 0:
                # Distances are calculated in the 640x480 YOLO space
                distances = cdist(current_yolo_coords, current_thermal_coords_scaled, metric='euclidean')

                all_possible_matches = []
                for y_idx in range(num_yolo):
                    for t_idx in range(num_thermal):
                        if distances[y_idx, t_idx] < matching_distance_threshold:
                            all_possible_matches.append((distances[y_idx, t_idx], y_idx, t_idx))
                
                all_possible_matches.sort(key=lambda x: x[0]) 

                for dist, y_idx, t_idx in all_possible_matches:
                    if y_idx not in matched_yolo_indices and t_idx not in matched_thermal_indices:
                        matched_yolo_indices.add(y_idx)
                        matched_thermal_indices.add(t_idx)
                        total_matched_distances.append(dist)
                        image_events_for_gui.append({
                            'timestamp': timestamp,
                            'type': 'TP',
                            'yolo_area': current_yolo_original_areas[y_idx],
                            'thermal_area': current_thermal_original_areas[t_idx],
                            'distance': dist
                        })
            
            # --- 4. Identify False Negatives and False Positives ---
            for y_idx in range(num_yolo):
                if y_idx not in matched_yolo_indices:
                    all_fn_yolo_areas.append(current_yolo_original_areas[y_idx])
                    image_events_for_gui.append({
                        'timestamp': timestamp,
                        'type': 'FN',
                        'yolo_area': current_yolo_original_areas[y_idx],
                        'thermal_area': None 
                    })

            for t_idx in range(num_thermal):
                if t_idx not in matched_thermal_indices:
                    all_fp_thermal_areas.append(current_thermal_original_areas[t_idx])
                    image_events_for_gui.append({
                        'timestamp': timestamp,
                        'type': 'FP',
                        'yolo_area': None,
                        'thermal_area': current_thermal_original_areas[t_idx]
                    })

        # --- 5. Calculate Overall Accuracy Metrics ---
        total_fns = len(all_fn_yolo_areas)
        total_fps = len(all_fp_thermal_areas)
        total_tps = len(total_matched_distances)

        total_relevant_events = total_tps + total_fps + total_fns 

        tp_percentage = (total_tps / total_relevant_events * 100) if total_relevant_events > 0 else 0
        fp_percentage = (total_fps / total_relevant_events * 100) if total_relevant_events > 0 else 0
        fn_percentage = (total_fns / total_relevant_events * 100) if total_relevant_events > 0 else 0

        precision = total_tps / (total_tps + total_fps) if (total_tps + total_fps) > 0 else 0
        recall = total_tps / (total_tps + total_fns) if (total_tps + total_fns) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        avg_pixel_offset = np.mean(total_matched_distances) if total_matched_distances else 0

        # --- 6. Correlation Analysis ---
        fn_yolo_distribution = None
        if all_fn_yolo_areas:
            fn_yolo_distribution = pd.cut(all_fn_yolo_areas, bins=yolo_area_bins, labels=yolo_area_bin_labels, right=False).value_counts().sort_index()

        fp_thermal_distribution = None
        if all_fp_thermal_areas:
            fp_thermal_distribution = pd.cut(all_fp_thermal_areas, bins=thermal_area_bins, labels=thermal_area_bin_labels, right=False).value_counts().sort_index()

        results = {
            'num_timestamps': len(df),
            'total_yolo_detections_considered': total_yolo_detections_considered,
            'total_thermal_detections_considered': total_thermal_detections_considered,
            'total_tps': total_tps,
            'total_fps': total_fps,
            'total_fns': total_fns,
            'tp_percentage': tp_percentage,
            'fp_percentage': fp_percentage,
            'fn_percentage': fn_percentage,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'avg_pixel_offset': avg_pixel_offset,
            'fn_yolo_distribution': fn_yolo_distribution,
            'fp_thermal_distribution': fp_thermal_distribution
        }
        
        return results, image_events_for_gui

# --- Main execution block ---
if __name__ == "__main__":
    current_script_dir = os.path.dirname(os.path.abspath(__file__))

    default_csv_filename = 'detections.csv'
    dummy_csv_filename = 'dummy_detections.csv'
    dummy_data_subdir = 'dummy_data'

    default_csv_path = os.path.join(current_script_dir, default_csv_filename)
    dummy_csv_path = os.path.join(current_script_dir, dummy_data_subdir, dummy_csv_filename)
    dummy_image_dir = os.path.join(current_script_dir, dummy_data_subdir)

    initial_csv_path = ""
    initial_image_dir = ""

    if os.path.exists(default_csv_path):
        initial_csv_path = default_csv_path
        initial_image_dir = current_script_dir 
    elif os.path.exists(dummy_csv_path):
        initial_csv_path = dummy_csv_path
        initial_image_dir = dummy_image_dir
        if not os.path.exists(initial_image_dir):
            messagebox.showwarning("Warning", 
                                   f"Dummy image directory not found at '{initial_image_dir}'. "
                                   "Image review might not work correctly. Please ensure dummy images are present.")
    else:
        # Initialize with empty paths, letting the user browse
        initial_csv_path = ""
        initial_image_dir = ""

    root = tk.Tk()
    app = AccuracyAnalyzerGUI(root)
    
    if initial_csv_path:
        app.csv_file_path.set(initial_csv_path)
    if initial_image_dir:
        app.image_dir_path.set(initial_image_dir)
    
    root.mainloop()
