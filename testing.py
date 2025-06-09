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


def test_mjpeg_reader():
    """Display MJPEG stream from ESP32-CAM."""
    reader = ESP32MJPEGReader("http://192.168.1.15/stream")
    print("🚀 Starting MJPEG reader test...")
    reader.start()

    print("⏳ Waiting for first frame...")
    start_time = time.time()
    while time.time() - start_time < 15:
        frame = reader.get_frame()
        if frame is not None:
            print(f"✅ Got first frame! Shape: {frame.shape}")
            break
        time.sleep(0.1)
    else:
        print("❌ No frame received within 15 seconds")
        reader.stop()
        return

    print("📺 Displaying stream. Press 'q' to quit.")
    frame_count = 0
    last_time = time.time()

    while True:
        frame = reader.get_frame()
        if frame is not None:
            now = time.time()
            fps = 1 / (now - last_time) if now > last_time else 0
            last_time = now

            cv2.putText(frame, f"Frame: {frame_count} | FPS: {fps:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow("ESP32-CAM MJPEG", frame)
            frame_count += 1

        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    reader.stop()
    cv2.destroyAllWindows()


def roboflow_integration():
    """FIXED: Send frames to Roboflow for inference."""
    reader = ESP32MJPEGReader("http://192.168.1.15/stream")
    print("🚀 Starting ESP32-CAM + Roboflow integration...")
    reader.start()

    print("⏳ Waiting for ESP32-CAM to be ready...")
    start_time = time.time()
    while time.time() - start_time < 10:
        frame = reader.get_frame()
        if frame is not None:
            print("✅ ESP32-CAM ready!")
            break
        time.sleep(0.1)
    else:
        print("❌ ESP32-CAM not ready")
        reader.stop()
        return

    print("📡 Processing frames with Roboflow (press 'q' to quit)...")
    last_inference = 0
    interval = 2.0

    # FIXED: Correct Roboflow workflow API endpoint and structure
    api_key = "PoT0jzGUtNb0bQOEf0Ja"
    workspace_id = "rov-crack-detection-3o91d"
    workflow_id = "custom-workflow"
    
    # Try the correct workflow endpoint format
    endpoint = f"https://api.roboflow.com/{workspace_id}/workflows/{workflow_id}/infer"

    while True:
        frame = reader.get_frame()
        if frame is not None:
            cv2.imshow("ESP32-CAM Live", frame)

            now = time.time()
            if now - last_inference > interval:
                last_inference = now
                try:
                    # FIXED: Proper base64 encoding without data URL prefix
                    _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                    image_base64 = base64.b64encode(buffer).decode('utf-8')

                    # FIXED: Correct workflow API payload structure
                    payload = {
                        "inputs": {
                            "image": {
                                "type": "base64",
                                "value": image_base64
                            }
                        }
                    }

                    headers = {
                        "Content-Type": "application/json"
                    }

                    # FIXED: Use query parameters for API key
                    params = {
                        "api_key": api_key
                    }

                    print("🔄 Sending request to Roboflow workflow...")
                    
                    try:
                        resp = requests.post(
                            endpoint,
                            json=payload,
                            headers=headers,
                            params=params,
                            timeout=30  # Increased timeout
                        )
                        
                        print(f"📊 Response status: {resp.status_code}")
                        
                        if resp.status_code == 200:
                            result = resp.json()
                            print(f"✅ Roboflow workflow success!")
                            print(f"📋 Full response: {result}")
                            
                            # FIXED: Better parsing of workflow outputs
                            if 'outputs' in result:
                                outputs = result['outputs']
                                print(f"🔍 Outputs found: {list(outputs.keys())}")
                                
                                for output_key, output_value in outputs.items():
                                    print(f"📝 Processing output: {output_key}")
                                    
                                    if isinstance(output_value, dict):
                                        if 'predictions' in output_value:
                                            predictions = output_value['predictions']
                                            print(f"🎯 Found {len(predictions)} predictions in {output_key}")
                                            
                                            for i, pred in enumerate(predictions):
                                                print(f"   Prediction {i+1}: {pred}")
                                        else:
                                            print(f"   {output_key} content: {output_value}")
                                    else:
                                        print(f"   {output_key}: {output_value}")
                            else:
                                print("⚠️ No 'outputs' field in response")
                                
                        elif resp.status_code == 422:
                            print("❌ Validation error - check your workflow configuration")
                            print(f"Response: {resp.text}")
                        elif resp.status_code == 401:
                            print("❌ Authentication error - check your API key")
                        elif resp.status_code == 404:
                            print("❌ Workflow not found - check workspace and workflow IDs")
                        else:
                            print(f"⚠️ Roboflow API error: {resp.status_code}")
                            print(f"Response: {resp.text}")
                            
                    except requests.exceptions.Timeout:
                        print("⚠️ Roboflow API timeout")
                    except requests.exceptions.ConnectionError:
                        print("⚠️ Roboflow API connection error")
                    except Exception as e:
                        print(f"⚠️ Roboflow API error: {e}")

                except Exception as e:
                    print(f"❌ Frame processing error: {e}")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    reader.stop()
    cv2.destroyAllWindows()


def roboflow_object_detection():
    """FIXED: Direct object detection model inference."""
    reader = ESP32MJPEGReader("http://192.168.1.15/stream")
    print("🚀 Starting ESP32-CAM + Roboflow Object Detection...")
    reader.start()

    print("⏳ Waiting for ESP32-CAM to be ready...")
    start_time = time.time()
    while time.time() - start_time < 10:
        frame = reader.get_frame()
        if frame is not None:
            print("✅ ESP32-CAM ready!")
            break
        time.sleep(0.1)
    else:
        print("❌ ESP32-CAM not ready")
        reader.stop()
        return

    print("📡 Processing frames with Roboflow Object Detection (press 'q' to quit)...")
    last_inference = 0
    interval = 2.0

    # FIXED: Correct object detection endpoint and parameters
    api_key = "PoT0jzGUtNb0bQOEf0Ja"
    model_endpoint = "https://detect.roboflow.com/underwater-crack-detection/3"

    while True:
        frame = reader.get_frame()
        if frame is not None:
            cv2.imshow("ESP32-CAM Live", frame)

            now = time.time()
            if now - last_inference > interval:
                last_inference = now
                try:
                    # FIXED: Encode frame properly
                    success, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                    if not success:
                        print("❌ Failed to encode frame")
                        continue
                        
                    image_base64 = base64.b64encode(buffer).decode('utf-8')

                    # FIXED: Correct parameters for object detection
                    params = {
                        "api_key": api_key,
                        "confidence": 0.40,
                        "overlap": 0.30
                    }

                    print("🔄 Sending request to Roboflow object detection...")

                    try:
                        resp = requests.post(
                            model_endpoint,
                            data=image_base64,
                            params=params,
                            headers={"Content-Type": "application/x-www-form-urlencoded"},
                            timeout=20
                        )
                        
                        print(f"📊 Response status: {resp.status_code}")
                        
                        if resp.status_code == 200:
                            result = resp.json()
                            predictions = result.get('predictions', [])
                            print(f"✅ Found {len(predictions)} crack detections")
                            
                            for i, pred in enumerate(predictions):
                                class_name = pred.get('class', 'unknown')
                                confidence = pred.get('confidence', 0)
                                x = pred.get('x', 0)
                                y = pred.get('y', 0)
                                width = pred.get('width', 0)
                                height = pred.get('height', 0)
                                print(f"   Detection {i+1}: {class_name} ({confidence:.2f}) at ({x:.0f},{y:.0f}) size:{width:.0f}x{height:.0f}")
                                
                        elif resp.status_code == 400:
                            print("❌ Bad request - check your image data")
                            print(f"Response: {resp.text}")
                        elif resp.status_code == 401:
                            print("❌ Authentication error - check your API key")
                        elif resp.status_code == 404:
                            print("❌ Model not found - check your model endpoint")
                        else:
                            print(f"⚠️ Object detection API error: {resp.status_code}")
                            print(f"Response: {resp.text}")
                            
                    except requests.exceptions.Timeout:
                        print("⚠️ Object detection API timeout")
                    except requests.exceptions.ConnectionError:
                        print("⚠️ Object detection API connection error")
                    except Exception as e:
                        print(f"⚠️ Object detection API error: {e}")

                except Exception as e:
                    print(f"❌ Frame processing error: {e}")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    reader.stop()
    cv2.destroyAllWindows()


# ADDED: Test individual components
def test_roboflow_api():
    """Test Roboflow API with a static image file."""
    print("🧪 Testing Roboflow API with static image...")
    
    # Create a simple test image
    test_image = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(test_image, "TEST IMAGE", (200, 240), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
    
    # Test object detection
    print("🔍 Testing object detection API...")
    success, buffer = cv2.imencode('.jpg', test_image, [cv2.IMWRITE_JPEG_QUALITY, 80])
    if success:
        image_base64 = base64.b64encode(buffer).decode('utf-8')
        
        api_key = "PoT0jzGUtNb0bQOEf0Ja"
        model_endpoint = "https://detect.roboflow.com/underwater-crack-detection/3"
        
        params = {
            "api_key": api_key,
            "confidence": 0.40,
            "overlap": 0.30
        }
        
        try:
            resp = requests.post(
                model_endpoint,
                data=image_base64,
                params=params,
                headers={"Content-Type": "application/x-www-form-urlencoded"},
                timeout=20
            )
            
            print(f"📊 Object Detection Response: {resp.status_code}")
            if resp.status_code == 200:
                result = resp.json()
                print(f"✅ API working! Response: {result}")
            else:
                print(f"❌ API Error: {resp.text}")
                
        except Exception as e:
            print(f"❌ API Test Error: {e}")
    
    # Test workflow
    print("🔄 Testing workflow API...")
    workspace_id = "rov-crack-detection-3o91d"
    workflow_id = "custom-workflow"
    endpoint = f"https://api.roboflow.com/{workspace_id}/workflows/{workflow_id}/infer"
    
    payload = {
        "inputs": {
            "image": {
                "type": "base64",
                "value": image_base64
            }
        }
    }
    
    try:
        resp = requests.post(
            endpoint,
            json=payload,
            params={"api_key": api_key},
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        
        print(f"📊 Workflow Response: {resp.status_code}")
        if resp.status_code == 200:
            result = resp.json()
            print(f"✅ Workflow working! Response: {result}")
        else:
            print(f"❌ Workflow Error: {resp.text}")
            
    except Exception as e:
        print(f"❌ Workflow Test Error: {e}")


if __name__ == "__main__":
    print("ESP32-CAM MJPEG Stream Handler")
    print("=" * 40)
    print("1. Test MJPEG reader only")
    print("2. Test with Roboflow Custom Workflow")
    print("3. Test with Roboflow Object Detection Model")
    print("4. Test Roboflow API (without camera)")

    choice = input("\nEnter choice (1, 2, 3, or 4): ").strip()
    if choice == "1":
        test_mjpeg_reader()
    elif choice == "2":
        roboflow_integration()
    elif choice == "3":
        roboflow_object_detection()
    elif choice == "4":
        test_roboflow_api()
    else:
        print("Invalid choice. Running MJPEG reader test by default...")
        test_mjpeg_reader()
    print("Done.")