# robot_server.py
import socket

HOST = "0.0.0.0"
PORT = 5005

print(f"[Robot] Ascult pe {HOST}:{PORT} ...")

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind((HOST, PORT))
    s.listen(1)
    conn, addr = s.accept()
    with conn:
        print(f"[Robot] Conectat cu {addr}")
        buffer = b""
        while True:
            data = conn.recv(1024)
            if not data:
                break
            buffer += data
            while b"\n" in buffer:
                line, buffer = buffer.split(b"\n", 1)
                cmd = line.decode("utf-8").strip()
                print(f"[Robot] Command received: {cmd}")
                # aici poți actualiza o interfață grafică, o simulare etc.
