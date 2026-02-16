from SerialComArduino import init_braccio, send_braccio, close_braccio

ser = init_braccio(port="COM8")

try:
    print("Enter servo angles: base shoulder elbow wrist_vert wrist_rot gripper")
    print("Example: 90 45 120 30 90 10")
    print("Type 'exit' to quit.")

    while True:
        user_input = input(">>> ")
        if user_input.lower() == "exit":
            break

        parts = user_input.strip().split()
        if len(parts) != 6:
            print("Please enter exact 6 values.")
            continue

        try:
            angles = [int(x) for x in parts]
        except ValueError:
            print("Invalid input. Please enter int only.")
            continue

        send_braccio(
            ser,
            base=angles[0],
            shoulder=angles[1],
            elbow=angles[2],
            wrist_vert=angles[3],
            wrist_rot=angles[4],
            gripper=angles[5]
        )

except KeyboardInterrupt:
    print("\n Exiting...")

finally:
    close_braccio(ser)
