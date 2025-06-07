import cv2
import numpy as np
import requests
import threading
import time

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
    reader = ESP32MJPEGReader("http://192.168.182.164/stream")
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
    """Send frames to Roboflow for inference."""
    reader = ESP32MJPEGReader("http://192.168.182.164/stream")
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

    # ✅ Correct Roboflow local inference URL
    endpoint = "http://localhost:9001/rov-crack-detection-3o91d/underwater-crack-detection/3?api_key=PoT0jzGUtNb0bQOEf0Ja"

    while True:
        frame = reader.get_frame()
        if frame is not None:
            cv2.imshow("ESP32-CAM Live", frame)

            now = time.time()
            if now - last_inference > interval:
                last_inference = now
                try:
                    _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                    image_data = buffer.tobytes()

                    try:
                        resp = requests.post(
                            endpoint,
                            files={"image": ("frame.jpg", image_data, "image/jpeg")},
                            timeout=10
                        )
                        if resp.ok:
                            print(f"✅ Roboflow result: {resp.json()}")
                        else:
                            print(f"⚠️ Failed: {endpoint} ({resp.status_code})")
                    except Exception as e:
                        print(f"⚠️ Error on {endpoint}: {e}")

                except Exception as e:
                    print(f"❌ Inference error: {e}")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    reader.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    print("ESP32-CAM MJPEG Stream Handler")
    print("=" * 40)
    print("1. Test MJPEG reader only")
    print("2. Test with Roboflow integration")

    choice = input("\nEnter choice (1 or 2): ").strip()
    if choice == "1":
        test_mjpeg_reader()
    elif choice == "2":
        roboflow_integration()
    else:
        print("Invalid choice. Running MJPEG reader test by default...")
        test_mjpeg_reader()
    print("Done.")