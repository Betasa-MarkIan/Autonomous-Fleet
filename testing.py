import cv2
import numpy as np
import requests
import threading
import time
import base64

class ESP32MJPEGReader:
    """MJPEG stream reader with reconnection logic for ESP32-CAM."""
    
    def __init__(self, url):
        self.url = url
        self.latest_frame = None
        self.running = False
        self.thread = None
        self.frame_lock = threading.Lock()

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._read_stream, daemon=True)
        self.thread.start()
        print("✅ ESP32-CAM MJPEG reader started")

    def _read_stream(self):
        retries = 0
        max_retries = 5

        while self.running and retries < max_retries:
            try:
                print(f"🔗 Connecting to MJPEG stream (attempt {retries + 1})...")
                response = requests.get(self.url, stream=True, timeout=(10, 30), 
                                        headers={'User-Agent': 'ESP32-CAM-Reader/1.0'})

                if response.status_code != 200:
                    print(f"❌ Failed to connect: {response.status_code}")
                    retries += 1
                    time.sleep(2)
                    continue

                print("✅ Connected to MJPEG stream")
                retries = 0
                buffer = b''
                last_data_time = time.time()

                for chunk in response.iter_content(chunk_size=1024):
                    if not self.running:
                        break

                    buffer += chunk
                    last_data_time = time.time()

                    if time.time() - last_data_time > 10:
                        print("⚠️ No data for 10 seconds, reconnecting...")
                        break

                    while True:
                        start = buffer.find(b'\xff\xd8')
                        end = buffer.find(b'\xff\xd9', start)

                        if start == -1 or end == -1:
                            break

                        jpeg_data = buffer[start:end+2]
                        buffer = buffer[end+2:]

                        try:
                            frame = cv2.imdecode(np.frombuffer(jpeg_data, dtype=np.uint8), cv2.IMREAD_COLOR)
                            if frame is not None:
                                with self.frame_lock:
                                    self.latest_frame = frame
                        except Exception as e:
                            print(f"⚠️ Frame decode error: {e}")

            except requests.exceptions.Timeout:
                print("⚠️ Read timeout, reconnecting...")
                retries += 1
            except requests.exceptions.ConnectionError:
                print("⚠️ Connection lost, reconnecting...")
                retries += 1
            except Exception as e:
                print(f"❌ Unexpected error: {e}")
                retries += 1

            time.sleep(2)

        if retries >= max_retries:
            print("❌ Failed after multiple attempts.")
        else:
            print("🔄 MJPEG reader stopped")

    def get_frame(self):
        with self.frame_lock:
            return self.latest_frame.copy() if self.latest_frame is not None else None

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=5)


def roboflow_crack_detection():
    """ESP32-CAM crack detection with Roboflow API and simplified measurements visualization."""
    
    # Configuration
    ESP32_CAM_URL = "http://192.168.1.15/stream"
    API_KEY = "PoT0jzGUtNb0bQOEf0Ja"
    MODEL_ENDPOINT = "https://detect.roboflow.com/underwater-crack-detection/3"
    
    # Calibration factor: pixels per millimeter
    # You need to adjust this based on your camera setup and distance to object
    # Example: if you know a 10mm object appears as 50 pixels, then PIXELS_PER_MM = 5.0
    PIXELS_PER_MM = 3.2  # Adjust this value based on your calibration
    
    # Initialize camera reader
    reader = ESP32MJPEGReader(ESP32_CAM_URL)
    print("🚀 Starting ESP32-CAM Crack Detection System...")
    print(f"📏 Calibration: {PIXELS_PER_MM} pixels per millimeter")
    reader.start()

    # Wait for camera to be ready
    print("⏳ Waiting for ESP32-CAM to be ready...")
    start_time = time.time()
    while time.time() - start_time < 10:
        frame = reader.get_frame()
        if frame is not None:
            print("✅ ESP32-CAM ready!")
            break
        time.sleep(0.1)
    else:
        print("❌ ESP32-CAM not ready within 10 seconds")
        reader.stop()
        return

    print("🔍 Starting crack detection with measurements... (press 'q' to quit)")
    
    # Detection variables
    last_inference = 0
    inference_interval = 2.0  # Process every 2 seconds
    latest_predictions = []   # Store latest detections for continuous display

    # Colors for different crack types (BGR format for OpenCV)
    colors = {
        'crack': (0, 0, 255),      # Red
        'defect': (0, 255, 255),   # Yellow
        'damage': (255, 0, 0),     # Blue
        'fissure': (255, 0, 255),  # Magenta
        'unknown': (255, 255, 255) # White
    }

    while True:
        frame = reader.get_frame()
        if frame is not None:
            display_frame = frame.copy()

            # Draw bounding boxes from latest predictions with compact measurements
            for pred in latest_predictions:
                class_name = pred.get('class', 'unknown')
                confidence = pred.get('confidence', 0)
                x = pred.get('x', 0)
                y = pred.get('y', 0)
                width = pred.get('width', 0)
                height = pred.get('height', 0)

                # Convert pixel dimensions to millimeters
                width_mm = width / PIXELS_PER_MM
                height_mm = height / PIXELS_PER_MM

                # Calculate bounding box coordinates (Roboflow uses center coordinates)
                x1 = int(x - width / 2)
                y1 = int(y - height / 2)
                x2 = int(x + width / 2)
                y2 = int(y + height / 2)

                # Get color for this detection
                color = colors.get(class_name.lower(), colors['unknown'])
                
                # Draw bounding box
                thickness = 2
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, thickness)

                # Create compact label with class name, confidence and measurements
                label = f"{class_name.upper()}: {confidence:.2f} | {width_mm:.1f}x{height_mm:.1f}mm"
                
                # Font settings
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.5
                font_thickness = 1
                
                # Get text size
                label_size = cv2.getTextSize(label, font, font_scale, font_thickness)[0]
                
                # Calculate label background position
                bg_x1 = x1
                bg_y1 = y1 - label_size[1] - 10
                bg_x2 = x1 + label_size[0] + 10
                bg_y2 = y1 - 2
                
                # Ensure background doesn't go off-screen
                if bg_y1 < 0:
                    bg_y1 = y2 + 2
                    bg_y2 = bg_y1 + label_size[1] + 8
                
                # Draw label background
                cv2.rectangle(display_frame, (bg_x1, bg_y1), (bg_x2, bg_y2), color, -1)
                cv2.rectangle(display_frame, (bg_x1, bg_y1), (bg_x2, bg_y2), (255, 255, 255), 1)

                # Draw label text
                text_y = bg_y1 + label_size[1] + 3
                cv2.putText(display_frame, label, (bg_x1 + 5, text_y),
                           font, font_scale, (255, 255, 255), font_thickness)

            # Display the frame
            cv2.imshow("ESP32-CAM Crack Detection", display_frame)

            # Perform AI inference at specified intervals
            current_time = time.time()
            if current_time - last_inference > inference_interval:
                last_inference = current_time
                
                try:
                    # Encode frame to base64 for API
                    success, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                    if not success:
                        print("❌ Failed to encode frame")
                        continue
                        
                    image_base64 = base64.b64encode(buffer).decode('utf-8')

                    # API parameters
                    params = {
                        "api_key": API_KEY,
                        "confidence": 0.40,  # Minimum confidence threshold
                        "overlap": 0.30      # Non-maximum suppression threshold
                    }

                    print(f"🔄 Analyzing frame for cracks...")

                    # Send request to Roboflow API
                    response = requests.post(
                        MODEL_ENDPOINT,
                        data=image_base64,
                        params=params,
                        headers={"Content-Type": "application/x-www-form-urlencoded"},
                        timeout=20
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        latest_predictions = result.get('predictions', [])
                        
                        if latest_predictions:
                            print(f"✅ Found {len(latest_predictions)} crack detection(s)")
                            for i, pred in enumerate(latest_predictions):
                                class_name = pred.get('class', 'unknown')
                                confidence = pred.get('confidence', 0)
                                width_px = pred.get('width', 0)
                                height_px = pred.get('height', 0)
                                width_mm = width_px / PIXELS_PER_MM
                                height_mm = height_px / PIXELS_PER_MM
                                print(f"   📏 Detection #{i+1}: {class_name} ({confidence:.2f}) "
                                      f"Size: {width_mm:.1f}mm × {height_mm:.1f}mm")
                        else:
                            print(f"ℹ️ No cracks detected")
                            
                    elif response.status_code == 400:
                        print("❌ Bad request - check image format")
                        latest_predictions = []
                    elif response.status_code == 401:
                        print("❌ Authentication error - check API key")
                        latest_predictions = []
                    elif response.status_code == 404:
                        print("❌ Model not found - check model endpoint")
                        latest_predictions = []
                    else:
                        print(f"⚠️ API error {response.status_code}: {response.text}")
                        latest_predictions = []
                        
                except requests.exceptions.Timeout:
                    print("⚠️ API request timeout")
                    latest_predictions = []
                except requests.exceptions.ConnectionError:
                    print("⚠️ API connection error")
                    latest_predictions = []
                except Exception as e:
                    print(f"❌ Error during inference: {e}")
                    latest_predictions = []

        # Check for quit command
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("👋 Stopping crack detection system...")
            break

    # Cleanup
    reader.stop()
    cv2.destroyAllWindows()
    print("✅ System stopped successfully")


if __name__ == "__main__":
    print("ESP32-CAM Crack Detection System with Measurements")
    print("=" * 50)
    print("📏 Real-time crack detection with millimeter measurements")
    print("📹 ESP32-CAM MJPEG stream processing")
    print("🤖 Powered by Roboflow AI")
    print()
    print("CALIBRATION REQUIRED:")
    print("Adjust PIXELS_PER_MM constant based on your setup:")
    print("1. Place an object of known size in the camera view")
    print("2. Measure its pixel dimensions on screen")
    print("3. Calculate: PIXELS_PER_MM = pixel_width / actual_mm")
    print()
    
    try:
        roboflow_crack_detection()
    except KeyboardInterrupt:
        print("\n⚠️ System interrupted by user")
    except Exception as e:
        print(f"\n❌ System error: {e}")
    
    print("Done.")
