import pandas as pd
import ast
import numpy as np
from scipy.spatial.distance import cdist
import tkinter as tk
from tkinter import filedialog, scrolledtext, ttk, messagebox
from PIL import Image, ImageTk # For image display in Tkinter
import os

# --- Adjustable Constants (Default Values) ---
# These will be used to initialize the GUI widgets.

# Camera Resolutions:
THERMAL_RAW_WIDTH_DEFAULT = 32
THERMAL_RAW_HEIGHT_DEFAULT = 24
YOLO_WIDTH_DEFAULT = 640
YOLO_HEIGHT_DEFAULT = 480

# Thermal FOV Adjustment (Cropping Factor):
THERMAL_CROP_FACTOR_DEFAULT = 0.8 

# Minimum Area Thresholds:
YOLO_MIN_AREA_THRESHOLD_DEFAULT = 500
THERMAL_MIN_AREA_THRESHOLD_DEFAULT = 10

# Matching Distance Threshold:
MATCHING_DISTANCE_THRESHOLD_DEFAULT = 20

# Area bins for correlation analysis:
YOLO_AREA_BINS_DEFAULT = [0, 2000, 5000, 15000, 30000, np.inf]
YOLO_AREA_BIN_LABELS_DEFAULT = ["<2k", "2k-5k", "5k-15k", "15k-30k", ">30k"]

THERMAL_AREA_BINS_DEFAULT = [0, 10, 20, 50, 100, np.inf]
THERMAL_AREA_BIN_LABELS_DEFAULT = ["<10", "10-20", "20-50", "50-100", ">100"]

class AccuracyAnalyzerGUI:
    def __init__(self, master):
        self.master = master
        master.title("Thermal Camera Accuracy Analyzer")
        master.geometry("1400x800") # Increased size for more content

        # --- Variables to hold GUI state ---
        self.csv_file_path = tk.StringVar(value="")
        self.image_dir_path = tk.StringVar(value="")

        self.thermal_crop_factor = tk.DoubleVar(value=THERMAL_CROP_FACTOR_DEFAULT)
        self.yolo_min_area_threshold = tk.IntVar(value=YOLO_MIN_AREA_THRESHOLD_DEFAULT)
        self.thermal_min_area_threshold = tk.IntVar(value=THERMAL_MIN_AREA_THRESHOLD_DEFAULT)
        self.matching_distance_threshold = tk.IntVar(value=MATCHING_DISTANCE_THRESHOLD_DEFAULT)

        # For bins, we'll use string variables and parse them
        self.yolo_area_bins_str = tk.StringVar(value=", ".join(map(str, YOLO_AREA_BINS_DEFAULT[:-1])) + ", inf")
        self.yolo_area_bin_labels_str = tk.StringVar(value=", ".join(YOLO_AREA_BIN_LABELS_DEFAULT))
        self.thermal_area_bins_str = tk.StringVar(value=", ".join(map(str, THERMAL_AREA_BINS_DEFAULT[:-1])) + ", inf")
        self.thermal_area_bin_labels_str = tk.StringVar(value=", ".join(THERMAL_AREA_BIN_LABELS_DEFAULT))

        # Results storage
        self.analysis_results = {}
        # Stores {'timestamp': ..., 'type': 'TP'/'FP'/'FN', 'yolo_area': ..., 'thermal_area': ..., 'distance': ...}
        self.image_events = [] 
        self.current_image_index = 0
        self.filtered_image_events = []
        self.image_display_filter = tk.StringVar(value="All") # "All", "TP", "FP", "FN"
        self.current_photo = None # To prevent garbage collection of image

        # --- Layout ---
        self.create_widgets()

    def create_widgets(self):
        # Main frames
        param_frame = ttk.LabelFrame(self.master, text="Parameters", padding="10")
        param_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

        results_frame = ttk.LabelFrame(self.master, text="Analysis Results", padding="10")
        results_frame.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")

        image_review_frame = ttk.LabelFrame(self.master, text="Image Review", padding="10")
        image_review_frame.grid(row=1, column=0, columnspan=2, padx=10, pady=10, sticky="nsew")

        self.master.grid_rowconfigure(0, weight=1)
        self.master.grid_rowconfigure(1, weight=1)
        self.master.grid_columnconfigure(0, weight=1)
        self.master.grid_columnconfigure(1, weight=1)

        # --- Parameter Frame Widgets ---
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

        ttk.Label(param_frame, text="Thermal Crop Factor (0.1-1.0):").grid(row=row_idx, column=0, sticky="w", pady=2)
        ttk.Scale(param_frame, from_=0.1, to=1.0, orient="horizontal", variable=self.thermal_crop_factor).grid(row=row_idx, column=1, sticky="ew", pady=2)
        ttk.Label(param_frame, textvariable=self.thermal_crop_factor, width=5).grid(row=row_idx, column=2, sticky="w", padx=5, pady=2)
        row_idx += 1

        ttk.Label(param_frame, text="YOLO Min Area Threshold:").grid(row=row_idx, column=0, sticky="w", pady=2)
        ttk.Entry(param_frame, textvariable=self.yolo_min_area_threshold, width=10).grid(row=row_idx, column=1, sticky="w", pady=2)
        row_idx += 1

        ttk.Label(param_frame, text="Thermal Min Area Threshold:").grid(row=row_idx, column=0, sticky="w", pady=2)
        ttk.Entry(param_frame, textvariable=self.thermal_min_area_threshold, width=10).grid(row=row_idx, column=1, sticky="w", pady=2)
        row_idx += 1

        ttk.Label(param_frame, text="Matching Distance Threshold:").grid(row=row_idx, column=0, sticky="w", pady=2)
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
        
        param_frame.grid_columnconfigure(1, weight=1) # Allow entry to expand

        # --- Results Frame Widgets ---
        self.results_text = scrolledtext.ScrolledText(results_frame, wrap="word", width=60, height=20, state="disabled")
        self.results_text.pack(expand=True, fill="both")

        # --- Image Review Frame Widgets ---
        # Controls for image filtering
        filter_frame = ttk.Frame(image_review_frame)
        filter_frame.pack(pady=5)
        ttk.Label(filter_frame, text="Show:").pack(side="left", padx=5)
        ttk.Radiobutton(filter_frame, text="All", variable=self.image_display_filter, value="All", command=self.apply_image_filter).pack(side="left", padx=5)
        ttk.Radiobutton(filter_frame, text="True Positives", variable=self.image_display_filter, value="TP", command=self.apply_image_filter).pack(side="left", padx=5)
        ttk.Radiobutton(filter_frame, text="False Positives", variable=self.image_display_filter, value="FP", command=self.apply_image_filter).pack(side="left", padx=5)
        ttk.Radiobutton(filter_frame, text="False Negatives", variable=self.image_display_filter, value="FN", command=self.apply_image_filter).pack(side="left", padx=5)

        # Image display area
        self.image_label = ttk.Label(image_review_frame, text="No image loaded.")
        self.image_label.pack(pady=10)
        
        self.image_info_label = ttk.Label(image_review_frame, text="Image Info: ")
        self.image_info_label.pack(pady=2)

        # Navigation buttons
        nav_frame = ttk.Frame(image_review_frame)
        nav_frame.pack(pady=5)
        ttk.Button(nav_frame, text="Previous", command=self.show_previous_image).pack(side="left", padx=10)
        ttk.Button(nav_frame, text="Next", command=self.show_next_image).pack(side="left", padx=10)

    def browse_csv(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if file_path:
            self.csv_file_path.set(file_path)
            # Automatically set image directory to the same folder as CSV
            self.image_dir_path.set(os.path.dirname(file_path))

    def browse_image_dir(self):
        dir_path = filedialog.askdirectory()
        if dir_path:
            self.image_dir_path.set(dir_path)

    def parse_bins(self, bins_str_var, labels_str_var, default_bins, default_labels):
        """Parses bin edges and labels from string variables."""
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

        # Parse bins from string inputs, using defaults on error
        yolo_bins, yolo_labels = self.parse_bins(self.yolo_area_bins_str, self.yolo_area_bin_labels_str, YOLO_AREA_BINS_DEFAULT, YOLO_AREA_BIN_LABELS_DEFAULT)
        thermal_bins, thermal_labels = self.parse_bins(self.thermal_area_bins_str, self.thermal_area_bin_labels_str, THERMAL_AREA_BINS_DEFAULT, THERMAL_AREA_BIN_LABELS_DEFAULT)

        try:
            results, image_events = self._calculate_accuracy(
                csv_path,
                self.thermal_crop_factor.get(),
                self.yolo_min_area_threshold.get(),
                self.thermal_min_area_threshold.get(),
                self.matching_distance_threshold.get(),
                yolo_bins, yolo_labels,
                thermal_bins, thermal_labels
            )
            self.analysis_results = results
            self.image_events = image_events
            self.display_results()
            self.apply_image_filter() # Initialize image review with current filter
        except Exception as e:
            messagebox.showerror("Analysis Error", f"An error occurred during analysis: {e}")

    def display_results(self):
        self.results_text.config(state="normal")
        self.results_text.delete(1.0, tk.END)
        
        res = self.analysis_results
        
        output = f"--- Thermal Camera Accuracy Analysis ---\n"
        output += f"Analysis based on {res['num_timestamps']} timestamps from '{self.csv_file_path.get()}'\n"
        output += f"YOLO Resolution: {YOLO_WIDTH_DEFAULT}x{YOLO_HEIGHT_DEFAULT}, Thermal Raw Resolution: {THERMAL_RAW_WIDTH_DEFAULT}x{THERMAL_RAW_HEIGHT_DEFAULT}\n"
        output += f"Thermal FOV Cropped to {self.thermal_crop_factor.get()*100:.0f}% central region.\n"
        output += f"YOLO Min Area Threshold: {self.yolo_min_area_threshold.get()} pixels, Thermal Min Area Threshold: {self.thermal_min_area_threshold.get()} pixels\n"
        output += f"Matching Distance Threshold: {self.matching_distance_threshold.get()} pixels (on {YOLO_WIDTH_DEFAULT}x{YOLO_HEIGHT_DEFAULT} grid)\n"
        output += "-" * 60 + "\n\n"

        output += f"Total YOLO Detections Considered (after area filter): {res['total_yolo_detections_considered']}\n"
        output += f"Total Thermal Detections Considered (after area/FOV filter): {res['total_thermal_detections_considered']}\n"
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
        output += f"   - Avg. Matched Distance Offset: {res['avg_pixel_offset']:.2f} pixels (for TP matches)\n"
        output += "-" * 60 + "\n\n"

        output += f"3. FALSE NEGATIVE ANALYSIS (YOLO detections missed by Thermal, by YOLO object size)\n"
        if res['fn_yolo_distribution'] is not None:
            output += str(res['fn_yolo_distribution']) + "\n"
            output += f"   Total FNs: {res['total_fns']}\n"
        else:
            output += "   No False Negatives recorded.\n"
        output += "-" * 60 + "\n\n"

        output += f"4. FALSE POSITIVE ANALYSIS (Thermal detections not matched by YOLO, by Thermal object size)\n"
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
            self.current_photo = None # Clear reference
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
                img.thumbnail((400, 300)) # Resize for display
                self.current_photo = ImageTk.PhotoImage(img) # Keep reference
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

    def _calculate_accuracy(self, csv_file, thermal_crop_factor, yolo_min_area_threshold, thermal_min_area_threshold,
                            matching_distance_threshold, yolo_area_bins, yolo_area_bin_labels,
                            thermal_area_bins, thermal_area_bin_labels):
        """
        Calculates the accuracy of a thermal camera by comparing its detections
        against a YOLO camera (considered the reference). It performs FOV alignment,
        size-based filtering, object matching, and analyzes False Positives/Negatives
        in relation to object size.
        Returns a dictionary of results and a list of image events for GUI display.
        """
        df = pd.read_csv(csv_file)
        
        # Convert CSV text representations of lists/dictionaries back to actual Python objects
        df['thermal_data'] = df['thermal_data'].apply(ast.literal_eval)
        df['yolo_data'] = df['yolo_data'].apply(ast.literal_eval)

        # Lists to store data for overall metrics and correlation analysis
        total_matched_distances = []
        all_fn_yolo_areas = []
        all_fp_thermal_areas = []
        
        # List to store image event details for GUI display
        image_events_for_gui = []

        # Counters for total detections considered after filtering
        total_yolo_detections_considered = 0
        total_thermal_detections_considered = 0

        # Pre-calculate Thermal FOV cropping and scaling parameters
        cropped_thermal_width = THERMAL_RAW_WIDTH_DEFAULT * thermal_crop_factor
        cropped_thermal_height = THERMAL_RAW_HEIGHT_DEFAULT * thermal_crop_factor
        
        offset_x = (THERMAL_RAW_WIDTH_DEFAULT - cropped_thermal_width) / 2
        offset_y = (THERMAL_RAW_HEIGHT_DEFAULT - cropped_thermal_height) / 2

        scale_x_thermal_to_yolo = YOLO_WIDTH_DEFAULT / cropped_thermal_width
        scale_y_thermal_to_yolo = YOLO_HEIGHT_DEFAULT / cropped_thermal_height

        # Iterate through each timestamp (row) in the DataFrame
        for index, row in df.iterrows():
            timestamp = row['timestamp']
            
            # --- 1. Process Thermal Detections: Filter, Crop, and Scale ---
            current_thermal_coords_scaled = []    # Centroids of thermal detections, scaled to YOLO grid
            current_thermal_original_areas = []   # Original areas of thermal detections (for FP analysis)
            
            for t_det in row['thermal_data']:
                # Apply minimum area threshold to filter out very small thermal detections
                if t_det['area'] < thermal_min_area_threshold:
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
                if yolo_area < yolo_min_area_threshold:
                    continue

                # Calculate the centroid of the YOLO bounding box
                cx = (yolo_x1 + yolo_x2) / 2
                cy = (yolo_y1 + yolo_y2) / 2
                
                current_yolo_coords.append([cx, cy])
                current_yolo_original_areas.append(yolo_area)
                
            total_yolo_detections_considered += len(current_yolo_coords)

            # --- 3. Perform Matching between Filtered Thermal and YOLO Detections ---
            num_thermal = len(current_thermal_coords_scaled)
            num_yolo = len(current_yolo_coords)

            matched_yolo_indices = set()
            matched_thermal_indices = set()

            if num_yolo > 0 and num_thermal > 0:
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
            
            # --- 4. Identify False Negatives (FNs) and False Positives (FPs) ---
            for y_idx in range(num_yolo):
                if y_idx not in matched_yolo_indices:
                    all_fn_yolo_areas.append(current_yolo_original_areas[y_idx])
                    image_events_for_gui.append({
                        'timestamp': timestamp,
                        'type': 'FN',
                        'yolo_area': current_yolo_original_areas[y_idx],
                        'thermal_area': None # No matching thermal detection
                    })

            for t_idx in range(num_thermal):
                if t_idx not in matched_thermal_indices:
                    all_fp_thermal_areas.append(current_thermal_original_areas[t_idx])
                    image_events_for_gui.append({
                        'timestamp': timestamp,
                        'type': 'FP',
                        'yolo_area': None, # No matching yolo detection
                        'thermal_area': current_thermal_original_areas[t_idx]
                    })

        # --- 5. Calculate Overall Accuracy Metrics ---
        
        total_fns = len(all_fn_yolo_areas)
        total_fps = len(all_fp_thermal_areas)
        total_tps = len(total_matched_distances)

        # Total relevant events for percentage calculation (TP + FP + FN)
        # This represents all instances where at least one camera detected something,
        # after filtering and FOV alignment, and we tried to classify it.
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

# --- Main execution block for GUI ---
if __name__ == "__main__":
    current_script_dir = os.path.dirname(os.path.abspath(__file__))

    # Define paths
    default_csv_filename = 'detections.csv'
    dummy_csv_filename = 'dummy_detections.csv'
    dummy_data_subdir = 'dummy_data'

    default_csv_path = os.path.join(current_script_dir, default_csv_filename)
    dummy_csv_path = os.path.join(current_script_dir, dummy_data_subdir, dummy_csv_filename)
    dummy_image_dir = os.path.join(current_script_dir, dummy_data_subdir)

    initial_csv_path = ""
    initial_image_dir = ""

    # Determine which CSV to use
    if os.path.exists(default_csv_path):
        initial_csv_path = default_csv_path
        initial_image_dir = current_script_dir # Assume images are in the same directory as detections.csv
    elif os.path.exists(dummy_csv_path):
        initial_csv_path = dummy_csv_path
        initial_image_dir = dummy_image_dir
        # Warn if the dummy image directory doesn't exist
        if not os.path.exists(initial_image_dir):
            messagebox.showwarning("Warning", 
                                   f"Dummy image directory not found at '{initial_image_dir}'. "
                                   "Image review might not work correctly. Please ensure dummy images are present.")
    else:
        messagebox.showerror("Error", 
                              f"Neither '{default_csv_filename}' (in script directory) "
                              f"nor '{dummy_csv_path}' found. "
                              "Please provide a CSV file using the 'Browse...' button.")
        # Initialize with empty paths, letting the user browse
        initial_csv_path = ""
        initial_image_dir = ""

    root = tk.Tk()
    app = AccuracyAnalyzerGUI(root)
    
    # Set initial paths for convenience if found
    if initial_csv_path:
        app.csv_file_path.set(initial_csv_path)
    if initial_image_dir:
        app.image_dir_path.set(initial_image_dir)
    
    root.mainloop()