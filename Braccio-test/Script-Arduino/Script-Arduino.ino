12#include <Braccio.h>
#include <Servo.h>

Servo base;
Servo shoulder;
Servo elbow;
Servo wrist_rot;
Servo wrist_ver;
Servo gripper;


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

  // Parse CSV
  char buf[64];
  line.toCharArray(buf, sizeof(buf));
  char* token = strtok(buf, ","); // skip "CMD"

  for (int i = 0; i < 6; i++) {
    token = strtok(NULL, ",");
    if (token) angles[i] = atoi(token);
  }

  // Move the Braccio arm
  Braccio.ServoMovement(
    100,          // speed
    angles[0],    // base
    angles[1],    // shoulder
    angles[2],    // elbow
    angles[3],    // wrist_rot
    angles[4],    // wrist_ver
    angles[5]     // gripper
  );
}
