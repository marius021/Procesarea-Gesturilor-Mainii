#!/usr/bin/env python3
"""Standalone test: sends one valid and one tampered command, prints Arduino replies."""
import hmac, hashlib, time, serial

SERIAL_PORT = "/dev/ttyACM0"   # adjust if your Pi enumerates the R4 differently
BAUD_RATE   = 115200
SECRET_KEY  = b"braccio-key-2026"

def sign_command(cmd: str) -> str:
    sig = hmac.new(SECRET_KEY, cmd.encode("ascii"), hashlib.sha256)
    return cmd + "|" + sig.hexdigest()[:8]

def send_line(ser, line: str):
    ser.reset_input_buffer()
    ser.write((line + "\n").encode("ascii"))
    time.sleep(0.3)
    resp = ser.readline().decode("ascii", errors="replace").strip()
    print(f"  SENT: {line}")
    print(f"  RECV: {resp}\n")
    return resp

def wait_for_ready(ser, timeout=10.0):
    """Block until the Arduino emits READY, draining the boot banner."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        line = ser.readline().decode("ascii", errors="replace").strip()
        if line == "READY":
            print("  Arduino READY\n")
            return True
        if line:
            print(f"  (startup) {line}")
    print("  WARNING: READY not received, continuing anyway\n")
    return False

def main():
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1.0)
    wait_for_ready(ser)          # replaces the fixed time.sleep(2.0)
    ser.reset_input_buffer()

    print("=== HMAC Serial Command Authentication Test ===\n")

    base_cmd = "CMD,90,95,45,90,90,24"

    # 1. Valid signed command
    signed = sign_command(base_cmd)
    print(f"[1] Valid command   : {signed}")
    print(f"    Arduino response : {send_line(ser, signed)}   <- accepted\n")

    # 2. Tampered payload, stale signature
    payload, sig = signed.rsplit("|", 1)
    tampered = "CMD,180,95,45,90,90,24|" + sig
    print(f"[2] Tampered command: {tampered}")
    print(f"    Arduino response : {send_line(ser, tampered)}   <- rejected\n")

    # 3. No signature at all
    print(f"[3] Unsigned command: {base_cmd}")
    print(f"    Arduino response : {send_line(ser, base_cmd)}   <- rejected\n")
    ser.close()

if __name__ == "__main__":
    main()