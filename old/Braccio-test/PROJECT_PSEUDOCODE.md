# Very Basic Project Pseudocode

This project uses a camera to detect a hand, converts simple hand features into
robot commands, and sends those commands to the Braccio robot through Arduino.

## Overall Flow

```text
START

initialize camera
initialize MediaPipe hand detector
initialize serial connection to Arduino / Braccio
set safe default robot pose

LOOP forever
    read one frame from camera
    flip frame for mirror-like control
    detect hand landmarks

    IF a hand is detected
        extract important hand points
        compute simple features from the hand
            palm vertical position
            thumb-index opening / closing

        map palm vertical position to elbow angle
        map thumb-index opening to gripper angle

        smooth movement to avoid sudden jumps
        limit step size for safety
        send servo command packet to Braccio

    ELSE
        move robot to neutral pose
        OR hold last safe pose

    show camera image with debug text

    IF user presses ESC
        break loop
END LOOP

close camera
close serial connection
END
```

## Current Simple Demo Logic

```text
START DEMO

fix base, shoulder, wrist vertical, wrist rotation
allow only elbow and gripper to change

LOOP
    get camera frame
    detect one hand

    IF hand exists
        compute palm_y from wrist + palm landmarks
        compute finger_angle from thumb tip, wrist, and index tip

        elbow_target = map hand height:
            hand high in image -> elbow up
            hand low in image -> elbow down

        gripper_target = map finger angle:
            fingers more open -> gripper more open
            fingers more closed -> gripper more closed

        smooth elbow_target and gripper_target
        rate-limit elbow_target and gripper_target
        send CMD packet to robot

    ELSE
        use failsafe behavior
            hold pose
            OR return to neutral
END LOOP
```

## Command Format

```text
CMD,base,shoulder,elbow,wrist_vertical,wrist_rotation,gripper
```
