import serial
import time

SERIAL_PORT = "COM8"      
BAUD_RATE   = 115200
TIMEOUT     = 1
UPDATE_DT   = 0.05

def init_braccio(port=SERIAL_PORT, baudrate=BAUD_RATE, timeout=TIMEOUT):
  
    ser = serial.Serial(port=port, baudrate=baudrate, timeout=timeout)
    time.sleep(2)  
    handshake(ser)
    return ser


def handshake(ser):
    ser.reset_input_buffer()
    ser.write(b"REQ\n")
    response = ser.readline().decode("utf-8").strip()
    if response != "ACK":
        raise RuntimeError("Handshake failed")
    else: print("Handshake successful")


def send_braccio(ser, base, shoulder, elbow, wrist_vert, wrist_rot, gripper):
  
    base       = max(0, min(180, base))
    shoulder   = max(15, min(165, shoulder))
    elbow      = max(0, min(180, elbow))
    wrist_vert = max(0, min(180, wrist_vert))
    wrist_rot  = max(0, min(180, wrist_rot))
    gripper    = max(10, min(73, gripper))

    frame = f"CMD,{base},{shoulder},{elbow},{wrist_vert},{wrist_rot},{gripper}\n"
    ser.write(frame.encode("utf-8"))


def close_braccio(ser):
    ser.close()
