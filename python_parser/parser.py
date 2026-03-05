import serial
import numpy as np
import cv2
import struct
import time

# --- Configuration ---
SERIAL_PORT = 'COM4'  # Change this to your serial port (e.g., 'COM3' on Windows)
BAUD_RATE = 57600
IMAGE_WIDTH = 32
IMAGE_HEIGHT = 24
DISPLAY_WIDTH = 320  # Resized width for display
DISPLAY_HEIGHT = 240 # Resized height for display

# --- Main Program ---
def main():
    try:
        # Initialize serial connection
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
        print(f"Connected to {SERIAL_PORT} at {BAUD_RATE} baud.")
    except serial.SerialException as e:
        print(f"Error opening serial port: {e}")
        return

    # Calculate scaling factors for drawing on the resized image
    scale_x = DISPLAY_WIDTH / IMAGE_WIDTH
    scale_y = DISPLAY_HEIGHT / IMAGE_HEIGHT

    # blank image now twice width & height
    blank_image = np.zeros((DISPLAY_HEIGHT * 2, DISPLAY_WIDTH * 2, 3), dtype=np.uint8)
    cv2.putText(blank_image, "Waiting for data...", (DISPLAY_WIDTH // 2, DISPLAY_HEIGHT),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.imshow("Thermal Detection", blank_image)

    while True:
        try:
            # read packet size
            size_bytes = ser.read(2)
            if len(size_bytes) != 2:
                print(f"Warning: Failed to read packet size. Resyncing...")
                ser.flushInput()
                continue
            packet_size = int.from_bytes(size_bytes, byteorder='little')

            packet_data = ser.read(packet_size)
            if len(packet_data) != packet_size:
                print(f"Warning: Read {len(packet_data)} bytes, expected {packet_size}. Resyncing...")
                ser.flushInput()
                continue

            # parse four consecutive images
            image_bytes  = packet_data[0:768]
            step1_bytes  = packet_data[768:1536]
            step2_bytes  = packet_data[1536:2304]
            step3_bytes  = packet_data[2304:3072]

            num_detected = packet_data[3072]
            person_data_start = 3073

            detected_people = []
            for i in range(num_detected):
                if person_data_start + 4 > len(packet_data):
                    print(f"Warning: Incomplete person data for person {i}. Resyncing...")
                    ser.flushInput()
                    detected_people = []
                    break
                y, x, area = struct.unpack('<BBH', packet_data[person_data_start:person_data_start+4])
                detected_people.append({'x': x, 'y': y, 'area': area})
                person_data_start += 4

            if len(detected_people) != num_detected:
                continue

            # convert to images
            def to_display(img_bytes):
                arr = np.frombuffer(img_bytes, dtype=np.uint8).reshape((IMAGE_HEIGHT, IMAGE_WIDTH))
                norm = cv2.normalize(arr, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                col = cv2.applyColorMap(norm, cv2.COLORMAP_INFERNO)
                return cv2.resize(col, (DISPLAY_WIDTH, DISPLAY_HEIGHT), interpolation=cv2.INTER_NEAREST)

            display_orig  = to_display(image_bytes)
            display_step1 = to_display(step1_bytes)
            display_step2 = to_display(step2_bytes)
            display_step3 = to_display(step3_bytes)

            cv2.putText(display_orig, "Raw Image", (10, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(display_step1, "Background Subtraction", (10, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(display_step2, "Morphological Opening", (10, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(display_step3, "Gaussian Blur", (10, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # draw detections on original only
            scale_x = DISPLAY_WIDTH / IMAGE_WIDTH
            scale_y = DISPLAY_HEIGHT / IMAGE_HEIGHT
            for person in detected_people:
                x_orig, y_orig, area = person['x'], person['y'], person['area']
                display_x = int(x_orig * scale_x)
                display_y = int(y_orig * scale_y)
                box_half = 5
                x1 = max(0, display_x - box_half)
                y1 = max(0, display_y - box_half)
                x2 = min(DISPLAY_WIDTH - 1, display_x + box_half)
                y2 = min(DISPLAY_HEIGHT - 1, display_y + box_half)

                cv2.rectangle(display_orig, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(display_orig, f"({x_orig},{y_orig})", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(display_orig, f"Area:{area}", (x1, y2 + 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # compose 2×2 grid
            top = np.hstack((display_orig, display_step1))
            bottom = np.hstack((display_step2, display_step3))
            combined = np.vstack((top, bottom))

            cv2.imshow("Thermal Detection", combined)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        except serial.SerialTimeoutException:
            print("Serial read timeout. No data received. Displaying blank image.")
            ser.flushInput()
            cv2.imshow("Thermal Detection", blank_image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        except Exception as e:
            print(f"An unexpected error occurred: {e}. Attempting to resync...")
            ser.flushInput()
            time.sleep(0.1)
            cv2.imshow("Thermal Detection", blank_image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # Clean up
    ser.close()
    cv2.destroyAllWindows()
    print("Serial connection closed and OpenCV windows destroyed.")

if __name__ == "__main__":
    main()