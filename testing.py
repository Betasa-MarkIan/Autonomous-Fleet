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
    """ESP32-CAM crack detection with Roboflow API and bounding boxes visualization."""
    
    # Configuration
    ESP32_CAM_URL = "http://192.168.1.15/stream"
    API_KEY = "PoT0jzGUtNb0bQOEf0Ja"
    MODEL_ENDPOINT = "https://detect.roboflow.com/underwater-crack-detection/3"
    
    # Initialize camera reader
    reader = ESP32MJPEGReader(ESP32_CAM_URL)
    print("🚀 Starting ESP32-CAM Crack Detection System...")
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

    print("🔍 Starting crack detection... (press 'q' to quit)")
    
    # Detection variables
    last_inference = 0
    inference_interval = 2.0  # Process every 2 seconds
    latest_predictions = []   # Store latest detections for continuous display
    frame_count = 0

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
            frame_count += 1
            display_frame = frame.copy()

            # Draw bounding boxes from latest predictions
            for i, pred in enumerate(latest_predictions):
                class_name = pred.get('class', 'unknown')
                confidence = pred.get('confidence', 0)
                x = pred.get('x', 0)
                y = pred.get('y', 0)
                width = pred.get('width', 0)
                height = pred.get('height', 0)

                # Calculate bounding box coordinates (Roboflow uses center coordinates)
                x1 = int(x - width / 2)
                y1 = int(y - height / 2)
                x2 = int(x + width / 2)
                y2 = int(y + height / 2)

                # Get color for this detection
                color = colors.get(class_name.lower(), colors['unknown'])
                
                # Draw bounding box with thickness based on confidence
                thickness = 3 if confidence > 0.7 else 2
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, thickness)

                # Draw corner markers for better visibility
                corner_length = 15
                cv2.line(display_frame, (x1, y1), (x1 + corner_length, y1), color, thickness + 1)
                cv2.line(display_frame, (x1, y1), (x1, y1 + corner_length), color, thickness + 1)
                cv2.line(display_frame, (x2, y1), (x2 - corner_length, y1), color, thickness + 1)
                cv2.line(display_frame, (x2, y1), (x2, y1 + corner_length), color, thickness + 1)
                cv2.line(display_frame, (x1, y2), (x1 + corner_length, y2), color, thickness + 1)
                cv2.line(display_frame, (x1, y2), (x1, y2 - corner_length), color, thickness + 1)
                cv2.line(display_frame, (x2, y2), (x2 - corner_length, y2), color, thickness + 1)
                cv2.line(display_frame, (x2, y2), (x2, y2 - corner_length), color, thickness + 1)

                # Create label with class name and confidence
                label = f"{class_name.upper()}: {confidence:.2f}"
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                
                # Draw label background
                cv2.rectangle(display_frame, (x1, y1 - label_size[1] - 15), 
                             (x1 + label_size[0] + 10, y1 - 5), color, -1)
                cv2.rectangle(display_frame, (x1, y1 - label_size[1] - 15), 
                             (x1 + label_size[0] + 10, y1 - 5), (255, 255, 255), 1)

                # Draw label text
                cv2.putText(display_frame, label, (x1 + 5, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                # Draw center point with ID number
                cv2.circle(display_frame, (int(x), int(y)), 4, color, -1)
                cv2.circle(display_frame, (int(x), int(y)), 6, (255, 255, 255), 1)
                cv2.putText(display_frame, f"#{i+1}", (int(x) + 8, int(y) + 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # Draw status information
            status_bg = np.zeros((80, 400, 3), dtype=np.uint8)
            cv2.putText(status_bg, f"Frame: {frame_count}", (10, 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(status_bg, f"Detections: {len(latest_predictions)}", (10, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0) if latest_predictions else (255, 255, 255), 2)
            
            # Overlay status on main frame
            display_frame[10:90, 10:410] = cv2.addWeighted(display_frame[10:90, 10:410], 0.3, status_bg, 0.7, 0)

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

                    print(f"🔄 Analyzing frame {frame_count} for cracks...")

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
                                x, y = pred.get('x', 0), pred.get('y', 0)
                                w, h = pred.get('width', 0), pred.get('height', 0)
                                area = w * h
                                print(f"   🎯 Detection #{i+1}: {class_name} ({confidence:.2f}) "
                                      f"at ({x:.0f},{y:.0f}) size:{w:.0f}x{h:.0f} area:{area:.0f}px²")
                        else:
                            print(f"ℹ️ No cracks detected in frame {frame_count}")
                            
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
    print("ESP32-CAM Crack Detection System")
    print("=" * 40)
    print("🎯 Real-time crack detection with bounding boxes")
    print("📹 ESP32-CAM MJPEG stream processing")
    print("🤖 Powered by Roboflow AI")
    print()
    
    try:
        roboflow_crack_detection()
    except KeyboardInterrupt:
        print("\n⚠️ System interrupted by user")
    except Exception as e:
        print(f"\n❌ System error: {e}")
    
    print("Done.")
