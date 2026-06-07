#include <Braccio.h>
#include <Servo.h>

Servo base;
Servo shoulder;
Servo elbow;
Servo wrist_rot;
Servo wrist_ver;
Servo gripper;

// ===== SAFE LIMITS (Conservative) =====
#define M1_MIN 0
#define M1_MAX 180

#define M2_MIN 15
#define M2_MAX 165

#define M3_MIN 0
#define M3_MAX 180

#define M4_MIN 0
#define M4_MAX 180

#define M5_MIN 0
#define M5_MAX 180

#define M6_MIN 10
#define M6_MAX 73   // Gripper safety limit

// ===== Clamp Function =====
int clamp(int value, int minVal, int maxVal) {
  if (value < minVal) return minVal;
  if (value > maxVal) return maxVal;
  return value;
}

void setup() {
  Serial.begin(115200);
  Braccio.begin();
}

void loop() {
  if (!Serial.available()) return;

  String line = Serial.readStringUntil('\n');
  line.trim();

  if (!line.startsWith("CMD,")) return;

  // Default angles
  int angles[6] = {90, 90, 90, 90, 90, 10};

  char buf[64];
  line.toCharArray(buf, sizeof(buf));
  char* token = strtok(buf, ","); // skip "CMD"

  for (int i = 0; i < 6; i++) {
    token = strtok(NULL, ",");
    if (token) angles[i] = atoi(token);
  }

  angles[0] = clamp(angles[0], M1_MIN, M1_MAX);
  angles[1] = clamp(angles[1], M2_MIN, M2_MAX);
  angles[2] = clamp(angles[2], M3_MIN, M3_MAX);
  angles[3] = clamp(angles[3], M4_MIN, M4_MAX);
  angles[4] = clamp(angles[4], M5_MIN, M5_MAX);
  angles[5] = clamp(angles[5], M6_MIN, M6_MAX);

  // Move the Braccio arm
  Braccio.ServoMovement(
    20,          // speed
    angles[0],    // base
    angles[1],    // shoulder
    angles[2],    // elbow
    angles[3],    // wrist_rot
    angles[4],    // wrist_ver
    angles[5]     // gripper
  );
}
