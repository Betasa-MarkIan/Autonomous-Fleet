#include <Arduino.h>
#include <WiFi.h>
#include <WebServer.h>
#include <HTTPClient.h>
#include <ESP32Servo.h>

// WiFi credentials
const char* ssid = "realme 11 Pro 5G"; 
const char* password = "12345678";

WebServer server(80);
Servo rudder;

// Relay and Button Variables
int relayPin = 19;
bool powerOn = false;

// Buzzer variables - Steering Signal
int buzzerPin = 27;
unsigned long lastBuzzerToggle = 0;
const unsigned long buzzerInterval = 500;
bool buzzerState = false;

// Sensor Timing
unsigned long lastSensorRead = 0;
const unsigned long sensorInterval = 1000;

// WiFi Timeout
unsigned long connectingStartAttempt = millis();
const unsigned long wifiTimeoutLimit = 30000;

// Distance Variables
long distanceA; long distanceB; long distanceC;

// Ultrasonic Pins+
#define ultraSonicTrigA 12
#define ultraSonicEchoA 14
#define ultraSonicTrigB 5
#define ultraSonicEchoB 18
#define ultraSonicTrigC 4
#define ultraSonicEchoC 16

// Distance Threshold and Rudder Angle for Steering
const float frontThreshold = 0.50;
const int rudderCenter = 90;
const int rudderLeft = 45;
const int rudderRight = 135;
String rudderDirection;

// Motor Control Pins
const int motorA1_A = 22;
const int motorA1_B = 26;

// PID Motor Control Variables
float desiredRPM = 100; 
float currentRPM = 0;
float error = 0;
float previousError = 0;
float integral = 0;
float derivative = 0;
float output = 0;

// PID Constants
float Kp = 1.0;
float Ki = 0.5;
float Kd = 0.1;

// PWM Channel Settings
const int pwmChannel = 0;
const int pwmFreq = 5000;
const int pwmResolution = 8; 

// Steering Flags and Timers
bool isSteering = false;
bool waitingToCenter = false;
unsigned long obstacleClearedTime = 0;
const unsigned long rudderReturnDelay = 2000;
unsigned long lastRudderUpdate = 0;
const unsigned long rudderUpdateInterval = 100;
float currentRudderAngle = rudderCenter; 

float readMotorRPM() {
  static float simulatedRPM = 0;
  simulatedRPM += (output - simulatedRPM) * 0.05;
  return simulatedRPM;
}

void setMotorSpeed(int rpm) {
  desiredRPM = rpm;
}

long getDistanceM(uint8_t trig, uint8_t echo) {
  digitalWrite(trig, LOW);
  delayMicroseconds(2);
  digitalWrite(trig, HIGH);
  delayMicroseconds(10);
  digitalWrite(trig, LOW);

  long duration = pulseIn(echo, HIGH, 30000);
  if (duration == 0) return 999;
  return (duration * 0.000344) / 2.0;
}

long pulseInNonBlocking(uint8_t pin, uint8_t state, unsigned long timeout) {
  unsigned long start = micros();
  while (digitalRead(pin) == state) {
    if (micros() - start >= timeout) return 0;
    yield();
  }
  while (digitalRead(pin) != state) {
    if (micros() - start >= timeout) return 0;
    yield();
  }
  unsigned long pulseStart = micros();
  while (digitalRead(pin) == state) {
    if (micros() - start >= timeout) return 0;
    yield();
  }
  return micros() - pulseStart;
}

void setup() {
  Serial.begin(115200);
  Serial.println("ESP32 Initialized");

  pinMode(relayPin, OUTPUT);
  digitalWrite(relayPin, LOW);

  pinMode(buzzerPin, OUTPUT);

  pinMode(ultraSonicTrigA, OUTPUT);
  pinMode(ultraSonicTrigB, OUTPUT);
  pinMode(ultraSonicTrigC, OUTPUT);
  pinMode(ultraSonicEchoA, INPUT);
  pinMode(ultraSonicEchoB, INPUT);
  pinMode(ultraSonicEchoC, INPUT);

  randomSeed(analogRead(34));
  rudder.attach(23);
  rudder.write(rudderCenter);

  pinMode(motorA1_A, OUTPUT);
  pinMode(motorA1_B, OUTPUT);
  ledcSetup(pwmChannel, pwmFreq, pwmResolution);
  ledcAttachPin(motorA1_A, pwmChannel);

  WiFi.begin(ssid, password);
  Serial.print("Connecting to WiFi");
  while (WiFi.status() != WL_CONNECTED && millis() - connectingStartAttempt < wifiTimeoutLimit) {
    delay(500);
    Serial.print(".");
  }

  if (WiFi.status() == WL_CONNECTED) {
    Serial.println("\nConnected to WiFi");
    Serial.print("IP address: ");
    Serial.println(WiFi.localIP());
    server.begin();
    Serial.println("Web server initialized.");
  } else {
    Serial.println("\nNo network detected.");
    Serial.println("Web server initialization failed.");
  }

  server.on("/toggle", HTTP_GET, []() {
    powerOn = !powerOn;
    digitalWrite(relayPin, powerOn ? HIGH : LOW);
    Serial.println(powerOn ? "Relay ON (via web)" : "Relay OFF (via web)");
    server.sendHeader("Access-Control-Allow-Origin", "*");
    server.send(200, "text/plain", powerOn ? "ON" : "OFF");
  });

  server.on("/toggle", HTTP_OPTIONS, []() {
    server.sendHeader("Access-Control-Allow-Origin", "*");
    server.sendHeader("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
    server.sendHeader("Access-Control-Allow-Headers", "Content-Type");
    server.send(204);
  });

  server.on("/status", HTTP_GET, []() {
    server.sendHeader("Access-Control-Allow-Origin", "*");
    server.send(200, "text/plain", powerOn ? "ON" : "OFF");
  });
}

void loop() {
  server.handleClient();

  if (powerOn && millis() - lastSensorRead >= sensorInterval) {
    lastSensorRead = millis();

    distanceA = getDistanceM(ultraSonicTrigA, ultraSonicEchoA);
    delay(50);
    distanceB = getDistanceM(ultraSonicTrigB, ultraSonicEchoB);
    delay(50);
    distanceC = getDistanceM(ultraSonicTrigC, ultraSonicEchoC);

    if (distanceA <= frontThreshold) {
      isSteering = true;
      waitingToCenter = false;

      if (millis() - lastRudderUpdate > rudderUpdateInterval) {
        lastRudderUpdate = millis();

        int targetAngle = (distanceB > distanceC) ? rudderLeft :
                          (distanceB < distanceC) ? rudderRight :
                          (random(0, 2) == 0 ? rudderLeft : rudderRight);

        rudderDirection = (targetAngle == rudderLeft) ? "Turning Left..." : "Turning Right...";

        float gain = 0.3;
        currentRudderAngle += (targetAngle - currentRudderAngle) * gain;
        rudder.write((int)currentRudderAngle);
        Serial.printf("Steering to: %d\n", (int)currentRudderAngle);
      }
    } else {
      rudderDirection = "Going straight...";

      if (isSteering) {
        if (millis() - lastBuzzerToggle >= buzzerInterval) {
          lastBuzzerToggle = millis();
          buzzerState = !buzzerState;
          digitalWrite(buzzerPin, buzzerState ? HIGH : LOW);
        }
      } else {
        digitalWrite(buzzerPin, LOW);
        buzzerState = false;
      }

      if (isSteering && !waitingToCenter) {
        obstacleClearedTime = millis();
        waitingToCenter = true;
      }

      if (waitingToCenter && (millis() - obstacleClearedTime >= rudderReturnDelay)) {
        if (millis() - lastRudderUpdate > rudderUpdateInterval) {
          lastRudderUpdate = millis();
          float gain = 0.3;
          currentRudderAngle += (rudderCenter - currentRudderAngle) * gain;
          rudder.write((int)currentRudderAngle);
          Serial.printf("Returning to center: %d\n", (int)currentRudderAngle);

          if (abs(currentRudderAngle - rudderCenter) < 2) {
            currentRudderAngle = rudderCenter;
            rudder.write(rudderCenter);
            isSteering = false;
            waitingToCenter = false;
            Serial.println("Rudder centered.");
          }
        }
      }
    }

    currentRPM = readMotorRPM();
    error = desiredRPM - currentRPM;
    integral += error;
    derivative = error - previousError;
    output = Kp * error + Ki * integral + Kd * derivative;

    int pwmValue = constrain((int)output, 0, 255);
    ledcWrite(pwmChannel, pwmValue);
    digitalWrite(motorA1_B, LOW);
    previousError = error;

    if (!powerOn) {
      rudder.write(rudderCenter);
      setMotorSpeed(0);
    } else if (isSteering) {
      setMotorSpeed(50);
    } else {
      setMotorSpeed(100);
    }

    static unsigned long lastPost = 0;
    int sendData;
    if (millis() - lastPost > 2000) {
      lastPost = millis();

      HTTPClient http;
      http.begin("http://192.168.62.210:8000/update-sensors");
      http.addHeader("Content-Type", "application/json");
      String json = "{";
      json += "\"frontDistance\":" + String(distanceA) + ",";
      json += "\"leftDistance\":" + String(distanceB) + ",";
      json += "\"rightDistance\":" + String(distanceC) + ",";
      json += "\"motorSpeed\":" + String(desiredRPM) + ",";
      json += "\"rudderDirection\":\"" + rudderDirection + "\"";
      json += "}";

      sendData = http.POST(json);

      if (sendData > 0) {
        String response = http.getString();
        Serial.println("Data sent successfully");
        Serial.println("Server response: " + response);
      } else {
        Serial.println("Failed to send data. HTTP code: " + String(sendData));
      }
      http.end();
    }
  }
}
