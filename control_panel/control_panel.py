import serial
import numpy as np
import cv2
import struct
import time
import csv
import os
import threading
import queue
import tkinter as tk
from tkinter import ttk, scrolledtext
from PIL import Image, ImageTk

# Attempt to import picamera2 and ultralytics.
# If they fail, provide a mock or skip functionality.
try:
    from picamera2 import Picamera2
    from ultralytics import YOLO
    HAS_CAMERA_YOLO = True
except ImportError:
    print("Warning: picamera2 or ultralytics not found. RGB+YOLO functionality will be disabled.")
    HAS_CAMERA_YOLO = False
    # Mock classes to prevent errors if not installed
    class MockPicamera2:
        def configure(self, *args, **kwargs): pass
        def create_preview_configuration(self, *args, **kwargs): return {}
        def start(self): pass
        def capture_array(self, *args, **kwargs): return np.zeros((480, 640, 3), dtype=np.uint8)
        def stop(self): pass
    class MockYOLO:
        def __init__(self, *args, **kwargs): pass
        def __call__(self, *args, **kwargs): return []
    Picamera2 = MockPicamera2
    YOLO = MockYOLO


# -----------------------------
# Configuration
# -----------------------------
PERSON_CLASS_ID = 0
FRAME_SIZE = (640, 480) # For RGB camera

SERIAL_PORT = '/dev/ttyUSB0' # Adjust for your system (e.g., 'COM3' on Windows)
BAUD_RATE = 57600

IMAGE_WIDTH = 32
IMAGE_HEIGHT = 24
DISPLAY_WIDTH = 320
DISPLAY_HEIGHT = 240

SAVE_DIR = "detections"
os.makedirs(SAVE_DIR, exist_ok=True)

SAVE_INTERVAL = 3          # seconds
CAPTURE_WINDOW = 5 * 60    # 5 minutes

# Default configurable parameters for the Node (should match serial_comms.cpp defaults)
DEFAULT_CONFIG = {
    "DT_BG_THRESHOLD": 30,
    "DT_MAX_DISTANCE": 255,
    "MIN_PERSON_AREA": 20,
    "MAX_PERSON_AREA": 200,
    "BG_FRAME_COUNT": 25,
    "TEMP_MIN": 10.0,
    "TEMP_MAX": 35.0,
}

# -----------------------------
# Utility Functions
# -----------------------------
def to_display(img_bytes):
    """Converts a byte array thermal image to a color-mapped OpenCV image for display."""
    arr = np.frombuffer(img_bytes, dtype=np.uint8).reshape((IMAGE_HEIGHT, IMAGE_WIDTH))
    arr = np.flip(arr, axis=1) # Flip horizontally to match typical camera orientation
    norm = cv2.normalize(arr, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    col = cv2.applyColorMap(norm, cv2.COLORMAP_INFERNO)
    return cv2.resize(col, (DISPLAY_WIDTH, DISPLAY_HEIGHT), interpolation=cv2.INTER_NEAREST)

def get_label_mask(labeled_image_bytes, label):
    """Extracts a binary mask for a specific label from the watershed output."""
    arr = np.frombuffer(labeled_image_bytes, dtype=np.uint8).reshape((IMAGE_HEIGHT, IMAGE_WIDTH))
    arr_flip = np.flip(arr, axis=1) # Flip horizontally
    return (arr_flip == label).astype(np.uint8) * 255

def save_image(img):
    """Saves an image to the SAVE_DIR with a timestamp."""
    timestamp = int(time.time())
    filename = os.path.join(SAVE_DIR, f"frame_{timestamp}.jpg")
    try:
        cv2.imwrite(filename, img)
    except Exception as e:
        print(f"Error saving image {filename}: {e}")

def save_csv(thermal_people, yolo_people):
    """Appends detection data to a CSV file."""
    timestamp = int(time.time())
    filename = os.path.join(SAVE_DIR, "detections.csv")
    file_exists = os.path.isfile(filename)

    try:
        with open(filename, "a", newline="") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["timestamp", "thermal_count", "yolo_count", "thermal_data", "yolo_data"])

            writer.writerow([
                timestamp,
                len(thermal_people),
                len(yolo_people),
                str(thermal_people), # Store list of dicts as string
                str(yolo_people)     # Store list of dicts as string
            ])
    except Exception as e:
        print(f"Error saving CSV {filename}: {e}")

# -----------------------------
# Serial Reader Thread
# -----------------------------
class SerialReaderThread(threading.Thread):
    def __init__(self, serial_port, baud_rate, data_queue, stop_event):
        super().__init__()
        self.serial_port = serial_port
        self.baud_rate = baud_rate
        self.data_queue = data_queue
        self.stop_event = stop_event
        self.ser = None
        self.HEADER = b'\xFE\x01\xFE\x01'
        self.last_sync_time = time.time()
        self.is_connected = False

    def run(self):
        try:
            self.ser = serial.Serial(self.serial_port, self.baud_rate, timeout=1)
            print(f"SerialReaderThread: Connected to {self.serial_port}")
            self.is_connected = True
        except serial.SerialException as e:
            print(f"SerialReaderThread: Serial error: {e}")
            self.is_connected = False
            self.data_queue.put({"status": f"Serial Error: {e}"})
            return

        while not self.stop_event.is_set():
            try:
                # Sync to header
                sync = b''
                while not self.stop_event.is_set():
                    byte = self.ser.read(1)
                    if not byte:
                        if time.time() - self.last_sync_time > 5:
                            print("SerialReaderThread: Still syncing...")
                            self.last_sync_time = time.time()
                        continue
                    sync += byte
                    sync = sync[-4:]
                    if sync == self.HEADER:
                        break
                    if time.time() - self.last_sync_time > 5:
                        print("SerialReaderThread: Still syncing...")
                        self.last_sync_time = time.time()
                
                if self.stop_event.is_set():
                    break

                # Read packet size
                size_bytes = self.ser.read(2)
                if len(size_bytes) != 2:
                    self.ser.flushInput()
                    continue

                packet_size = int.from_bytes(size_bytes, "little")
                packet_data = self.ser.read(packet_size)
                if len(packet_data) != packet_size:
                    self.ser.flushInput()
                    continue

                # Parse packet
                image_bytes  = packet_data[0:768]
                step1_bytes  = packet_data[768:1536]
                step2_bytes  = packet_data[1536:2304]
                step3_bytes  = packet_data[2304:3072]

                num_detected = packet_data[3072]
                idx_ptr = 3073

                thermal_people = []
                for _ in range(num_detected):
                    if idx_ptr + 4 > len(packet_data):
                        print("SerialReaderThread: Packet data truncated for thermal people.")
                        thermal_people = []
                        break
                    y, x, area = struct.unpack('<BBH', packet_data[idx_ptr:idx_ptr+4])
                    thermal_people.append({"x": x, "y": y, "area": area})
                    idx_ptr += 4
                
                # Put data into queue for GUI to process
                self.data_queue.put({
                    "images": {
                        "orig": image_bytes,
                        "bg": step1_bytes,
                        "dist_map": step2_bytes,
                        "watershed": step3_bytes
                    },
                    "thermal_people": thermal_people,
                    "status": "Receiving data..."
                })

            except serial.SerialException as e:
                print(f"SerialReaderThread: Serial communication error: {e}")
                self.is_connected = False
                self.data_queue.put({"status": f"Serial Comm Error: {e}"})
                time.sleep(1) # Wait before retrying
            except Exception as e:
                print(f"SerialReaderThread: Error processing packet: {e}")
                self.ser.flushInput() # Clear buffer on error
                self.data_queue.put({"status": f"Packet Error: {e}"})

        if self.ser:
            self.ser.close()
            print("SerialReaderThread: Serial port closed.")

    def write_serial(self, data):
        """Thread-safe method to write data to the serial port."""
        if self.ser and self.ser.is_open:
            try:
                self.ser.write(data)
                return True
            except serial.SerialException as e:
                print(f"SerialReaderThread: Error writing to serial: {e}")
                return False
        return False

# -----------------------------
# Tkinter GUI Application
# -----------------------------
class ControlPanelApp:
    def __init__(self, master):
        self.master = master
        master.title("CrowdAware Control Panel")
        master.geometry("1400x900") # Adjust as needed
        master.protocol("WM_DELETE_WINDOW", self.on_closing)

        self.data_queue = queue.Queue()
        self.stop_event = threading.Event()

        self.yolo_model = None
        self.picam2 = None
        self.has_camera_yolo = HAS_CAMERA_YOLO

        self.init_camera_yolo()
        self.init_serial_thread()
        self.create_widgets()
        
        self.last_save_time = 0
        self.capture_start_time = None

        self.update_gui() # Start the GUI update loop

    def init_camera_yolo(self):
        if self.has_camera_yolo:
            print("Initializing YOLO model...")
            self.yolo_model = YOLO("yolo26n.onnx") # Ensure this file is present
            print("Initializing Picamera2...")
            self.picam2 = Picamera2()
            self.picam2.configure(
                self.picam2.create_preview_configuration(
                    {"format": "RGB888", "size": FRAME_SIZE}
                )
            )
            self.picam2.start()
            print("Picamera2 started.")
        else:
            print("RGB+YOLO functionality disabled due to missing libraries.")

    def init_serial_thread(self):
        self.serial_thread = SerialReaderThread(SERIAL_PORT, BAUD_RATE, self.data_queue, self.stop_event)
        self.serial_thread.daemon = True # Allow main program to exit even if thread is running
        self.serial_thread.start()
        self.master.after(100, self.check_serial_connection) # Check connection status shortly after starting

    def check_serial_connection(self):
        if not self.serial_thread.is_connected:
            self.status_label.config(text=f"Status: Serial not connected to {SERIAL_PORT}. Please check connection and restart.", foreground="red")
        else:
            self.status_label.config(text=f"Status: Connected to {SERIAL_PORT}", foreground="green")

    def create_widgets(self):
        # Main frames
        self.main_frame = ttk.Frame(self.master, padding="10")
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        self.image_frame = ttk.Frame(self.main_frame)
        self.image_frame.grid(row=0, column=0, rowspan=2, padx=10, pady=10, sticky="nsew")

        self.config_frame = ttk.LabelFrame(self.main_frame, text="Node Configuration", padding="10")
        self.config_frame.grid(row=0, column=1, padx=10, pady=10, sticky="new")

        self.log_frame = ttk.LabelFrame(self.main_frame, text="Detection Log", padding="10")
        self.log_frame.grid(row=1, column=1, padx=10, pady=10, sticky="nsew")

        self.status_frame = ttk.Frame(self.main_frame)
        self.status_frame.grid(row=2, column=0, columnspan=2, pady=5, sticky="ew")

        # Configure grid weights for resizing
        self.main_frame.grid_columnconfigure(0, weight=3) # Image frame takes more space
        self.main_frame.grid_columnconfigure(1, weight=1)
        self.main_frame.grid_rowconfigure(1, weight=1) # Log frame expands vertically

        # Image Display
        self.thermal_labels = {}
        image_titles = ["Raw Thermal", "Background", "Distance Map", "Watershed"]
        for i, title in enumerate(image_titles):
            row, col = divmod(i, 2)
            frame = ttk.Frame(self.image_frame, borderwidth=1, relief="solid")
            frame.grid(row=row, column=col, padx=5, pady=5, sticky="nsew")
            ttk.Label(frame, text=title).pack(side=tk.TOP)
            label = ttk.Label(frame)
            label.pack(side=tk.BOTTOM)
            self.thermal_labels[title.lower().replace(' ', '_')] = label
            self.image_frame.grid_columnconfigure(col, weight=1)
            self.image_frame.grid_rowconfigure(row, weight=1)

        self.rgb_yolo_label_frame = ttk.Frame(self.image_frame, borderwidth=1, relief="solid")
        self.rgb_yolo_label_frame.grid(row=0, column=2, rowspan=2, padx=5, pady=5, sticky="nsew")
        ttk.Label(self.rgb_yolo_label_frame, text="RGB + YOLO").pack(side=tk.TOP)
        self.rgb_yolo_label = ttk.Label(self.rgb_yolo_label_frame)
        self.rgb_yolo_label.pack(side=tk.BOTTOM)
        self.image_frame.grid_columnconfigure(2, weight=2) # RGB image larger

        # Configuration Controls
        self.config_vars = {}
        config_params = [
            ("DT_BG_THRESHOLD", "int"), ("DT_MAX_DISTANCE", "int"),
            ("MIN_PERSON_AREA", "int"), ("MAX_PERSON_AREA", "int"),
            ("BG_FRAME_COUNT", "int"),
            ("TEMP_MIN", "float"), ("TEMP_MAX", "float")
        ]
        for i, (param, p_type) in enumerate(config_params):
            ttk.Label(self.config_frame, text=param + ":").grid(row=i, column=0, sticky="w", pady=2)
            var = tk.StringVar(value=str(DEFAULT_CONFIG[param]))
            entry = ttk.Entry(self.config_frame, textvariable=var, width=15)
            entry.grid(row=i, column=1, sticky="ew", pady=2, padx=5)
            self.config_vars[param] = (var, p_type)
        
        ttk.Button(self.config_frame, text="Apply Configuration", command=self.send_config).grid(row=len(config_params), column=0, columnspan=2, pady=10)

        # Detection Log
        self.log_tree = ttk.Treeview(self.log_frame, columns=("Timestamp", "Thermal Count", "YOLO Count", "Thermal Data", "YOLO Data"), show="headings")
        self.log_tree.heading("Timestamp", text="Timestamp")
        self.log_tree.heading("Thermal Count", text="Thermal Count")
        self.log_tree.heading("YOLO Count", text="YOLO Count")
        self.log_tree.heading("Thermal Data", text="Thermal Data")
        self.log_tree.heading("YOLO Data", text="YOLO Data")

        self.log_tree.column("Timestamp", width=100, anchor="center")
        self.log_tree.column("Thermal Count", width=80, anchor="center")
        self.log_tree.column("YOLO Count", width=80, anchor="center")
        self.log_tree.column("Thermal Data", width=200, anchor="w")
        self.log_tree.column("YOLO Data", width=200, anchor="w")

        self.log_tree.pack(fill=tk.BOTH, expand=True)

        log_scroll = ttk.Scrollbar(self.log_frame, orient="vertical", command=self.log_tree.yview)
        self.log_tree.configure(yscrollcommand=log_scroll.set)
        log_scroll.pack(side=tk.RIGHT, fill=tk.Y)


        # Status Bar
        self.status_label = ttk.Label(self.status_frame, text="Status: Initializing...", anchor="w")
        self.status_label.pack(fill=tk.X, expand=True)

        self.tk_images = {} # To hold PhotoImage references

    def send_config(self):
        if not self.serial_thread.is_connected:
            self.status_label.config(text="Status: Not connected to Node. Cannot send config.", foreground="red")
            return

        commands = []
        for param, (var, p_type) in self.config_vars.items():
            try:
                value = var.get()
                if p_type == "int":
                    int(value) # Validate
                elif p_type == "float":
                    float(value) # Validate
                commands.append(f"SET_{param}={value}\n")
            except ValueError:
                self.status_label.config(text=f"Status: Invalid value for {param}. Must be {p_type}.", foreground="red")
                return
        
        for cmd in commands:
            self.serial_thread.write_serial(cmd.encode('ascii'))
            time.sleep(0.01) # Small delay between commands

        self.status_label.config(text="Status: Configuration sent to Node.", foreground="blue")
        print("Sent configuration commands:", commands)

    def update_gui(self):
        try:
            while not self.data_queue.empty():
                data = self.data_queue.get_nowait()
                
                if "status" in data:
                    self.status_label.config(text=f"Status: {data['status']}", foreground="black")
                    continue

                images = data["images"]
                thermal_people = data["thermal_people"]

                # Process thermal images
                display_orig  = to_display(images["orig"])
                display_step1 = to_display(images["bg"])
                display_step2 = to_display(images["dist_map"])
                display_step3 = to_display(images["watershed"])

                scale_x = DISPLAY_WIDTH / IMAGE_WIDTH
                scale_y = DISPLAY_HEIGHT / IMAGE_HEIGHT

                # Overlay thermal detections on raw image
                for idx, p in enumerate(thermal_people):
                    label = idx + 1
                    mask_small = get_label_mask(images["watershed"], label)
                    mask_large = cv2.resize(mask_small, (DISPLAY_WIDTH, DISPLAY_HEIGHT), interpolation=cv2.INTER_NEAREST)

                    overlay = display_orig.copy()
                    overlay[mask_large > 0] = (0, 255, 0) # Green overlay
                    display_orig = cv2.addWeighted(display_orig, 0.7, overlay, 0.3, 0)

                    # Centroid coordinates on the flipped image
                    # The Node sends (x,y) where x is 0-31 and y is 0-23.
                    # We flipped the image, so x-coordinate needs to be adjusted for display.
                    # Original x was 0 (left) to 31 (right). Flipped, 0 (right) to 31 (left).
                    # So, (31 - p["x"]) maps the original x to the flipped x for display.
                    tx = int((IMAGE_WIDTH - 1 - p["x"]) * scale_x) # Adjust for flipped image
                    ty = int(p["y"] * scale_y)

                    cv2.putText(display_orig, f"({p['x']},{p['y']})", (tx, ty),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)
                    cv2.putText(display_orig, f"Area:{p['area']}", (tx, ty+20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)

                cv2.putText(display_orig, "Raw Image", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(display_step1, "Background", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(display_step2, "Distance Map", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(display_step3, "Watershed", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                # Update thermal image labels
                self._update_image_label(self.thermal_labels["raw_thermal"], display_orig)
                self._update_image_label(self.thermal_labels["background"], display_step1)
                self._update_image_label(self.thermal_labels["distance_map"], display_step2)
                self._update_image_label(self.thermal_labels["watershed"], display_step3)

                # RGB + YOLO
                rgb_frame = None
                yolo_people = []
                if self.has_camera_yolo and self.picam2:
                    rgb_frame = self.picam2.capture_array("main")
                    results = self.yolo_model(rgb_frame, imgsz=320, verbose=False)

                    for r in results:
                        for box in r.boxes:
                            if int(box.cls[0]) == PERSON_CLASS_ID:
                                x1, y1, x2, y2 = map(int, box.xyxy[0])
                                yolo_people.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2})
                                cv2.rectangle(rgb_frame, (x1, y1), (x2, y2), (0,255,0), 2)
                    
                    cv2.putText(rgb_frame, "RGB + YOLO", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    rgb_resized = cv2.resize(rgb_frame, (DISPLAY_WIDTH * 2 + 10, DISPLAY_HEIGHT * 2 + 10)) # Match thermal grid size for combined display
                    self._update_image_label(self.rgb_yolo_label, rgb_resized)
                else:
                    # Display a placeholder if camera/YOLO is not available
                    placeholder = np.zeros((FRAME_SIZE[1], FRAME_SIZE[0], 3), dtype=np.uint8)
                    cv2.putText(placeholder, "RGB+YOLO Disabled", (50, FRAME_SIZE[1]//2), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
                    rgb_resized = cv2.resize(placeholder, (DISPLAY_WIDTH * 2 + 10, DISPLAY_HEIGHT * 2 + 10))
                    self._update_image_label(self.rgb_yolo_label, rgb_resized)

                # Update log
                timestamp_str = time.strftime("%H:%M:%S", time.localtime())
                self.log_tree.insert("", "end", values=(timestamp_str, len(thermal_people), len(yolo_people), str(thermal_people), str(yolo_people)))
                self.log_tree.yview_moveto(1) # Scroll to bottom

                # Saving logic
                now = time.time()
                detected = (len(thermal_people) > 0) or (len(yolo_people) > 0)

                if detected and self.capture_start_time is None:
                    self.capture_start_time = now
                    print("Capture window started.")

                if self.capture_start_time is not None:
                    if now - self.capture_start_time <= CAPTURE_WINDOW:
                        if now - self.last_save_time >= SAVE_INTERVAL:
                            if rgb_frame is not None: # Only save if RGB frame was captured
                                combined_img = np.hstack((rgb_resized, np.vstack([np.hstack([display_orig, display_step1]), np.hstack([display_step2, display_step3])])))
                                save_image(combined_img)
                            save_csv(thermal_people, yolo_people)
                            self.last_save_time = now
                    else:
                        self.capture_start_time = None
                        print("Capture window ended.")
                
                # Always save CSV every SAVE_INTERVAL even outside capture window
                if now - self.last_save_time >= SAVE_INTERVAL:
                    save_csv(thermal_people, yolo_people)
                    self.last_save_time = now

        except Exception as e:
            print(f"Error in update_gui: {e}")
            self.status_label.config(text=f"Status: GUI Error: {e}", foreground="red")
        
        self.master.after(50, self.update_gui) # Schedule next update

    def _update_image_label(self, label_widget, cv_image):
        """Helper to convert OpenCV image to PhotoImage and update a Tkinter Label."""
        rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_image)
        
        # Store PhotoImage in a dictionary to prevent garbage collection
        # Use the label widget itself as the key or a unique identifier
        self.tk_images[label_widget] = ImageTk.PhotoImage(image=pil_image)
        label_widget.config(image=self.tk_images[label_widget])

    def on_closing(self):
        print("Closing application...")
        self.stop_event.set() # Signal serial thread to stop
        self.serial_thread.join(timeout=2) # Wait for thread to finish
        if self.serial_thread.is_alive():
            print("Warning: Serial thread did not terminate gracefully.")

        if self.picam2:
            self.picam2.stop()
            print("Picamera2 stopped.")
        
        self.master.destroy()

# -----------------------------
# Main Execution
# -----------------------------
if __name__ == "__main__":
    root = tk.Tk()
    app = ControlPanelApp(root)
    root.mainloop()