#include <Braccio.h>
#include <Servo.h>
#include <string.h>

namespace {

const unsigned long BAUD_RATE = 115200;
const unsigned long COMMAND_TIMEOUT_MS = 1000;
const int SERVO_STEP_DELAY_MS = 20;

const int M1_MIN = 0;
const int M1_MAX = 180;
const int M2_MIN = 15;
const int M2_MAX = 165;
const int M3_MIN = 15;
const int M3_MAX = 165;
const int M4_MIN = 0;
const int M4_MAX = 180;
const int M5_MIN = 0;
const int M5_MAX = 180;
const int M6_MIN = 10;
const int M6_MAX = 73;

const int NEUTRAL[6] = {90, 95, 95, 90, 90, 24};

char inputBuffer[64];
size_t inputLength = 0;
unsigned long lastCommandAt = 0;
bool inNeutral = false;

int clampJoint(int value, int minValue, int maxValue) {
  if (value < minValue) return minValue;
  if (value > maxValue) return maxValue;
  return value;
}

void sanitizeAngles(int angles[6]) {
  angles[0] = clampJoint(angles[0], M1_MIN, M1_MAX);
  angles[1] = clampJoint(angles[1], M2_MIN, M2_MAX);
  angles[2] = clampJoint(angles[2], M3_MIN, M3_MAX);
  angles[3] = clampJoint(angles[3], M4_MIN, M4_MAX);
  angles[4] = clampJoint(angles[4], M5_MIN, M5_MAX);
  angles[5] = clampJoint(angles[5], M6_MIN, M6_MAX);
}

void moveArm(const int angles[6]) {
  Braccio.ServoMovement(
    SERVO_STEP_DELAY_MS,
    angles[0],
    angles[1],
    angles[2],
    angles[3],  // wrist_vertical
    angles[4],  // wrist_rotation
    angles[5]
  );
}

bool parseCommand(char* line, int outAngles[6]) {
  if (strncmp(line, "CMD,", 4) != 0) {
    return false;
  }

  char* token = strtok(line, ",");
  for (int i = 0; i < 6; ++i) {
    token = strtok(NULL, ",");
    if (token == nullptr) {
      return false;
    }
    outAngles[i] = atoi(token);
  }
  return true;
}

void handleLine(char* line) {
  int angles[6];
  if (!parseCommand(line, angles)) {
    Serial.println("ERR");
    return;
  }

  sanitizeAngles(angles);
  moveArm(angles);
  lastCommandAt = millis();
  inNeutral = false;
  Serial.println("OK");
}

void readSerial() {
  while (Serial.available() > 0) {
    const char c = static_cast<char>(Serial.read());

    if (c == '\r') {
      continue;
    }

    if (c == '\n') {
      inputBuffer[inputLength] = '\0';
      if (inputLength > 0) {
        handleLine(inputBuffer);
      }
      inputLength = 0;
      continue;
    }

    if (inputLength < sizeof(inputBuffer) - 1) {
      inputBuffer[inputLength++] = c;
    } else {
      inputLength = 0;
      Serial.println("ERR");
    }
  }
}

void watchdogNeutral() {
  const unsigned long now = millis();
  if (!inNeutral && lastCommandAt != 0 && now - lastCommandAt > COMMAND_TIMEOUT_MS) {
    int neutralPose[6];
    memcpy(neutralPose, NEUTRAL, sizeof(NEUTRAL));
    moveArm(neutralPose);
    inNeutral = true;
  }
}

}  // namespace

void setup() {
  Serial.begin(BAUD_RATE);
  Braccio.begin();
  delay(500);
  moveArm(NEUTRAL);
  lastCommandAt = millis();
  inNeutral = true;
  Serial.println("READY");
}

void loop() {
  readSerial();
  watchdogNeutral();
}
