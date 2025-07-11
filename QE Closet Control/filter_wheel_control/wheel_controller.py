"""
Needs 'FWxC_COMMAND_LIB.py' and 'FilterWheel102_win64.dll'
64-bit version
"""
import sys
from FWxC_COMMAND_LIB import (
    FWxCListDevices,
    FWxCOpen,
    FWxCGetPosition,
    FWxCSetPosition,
    FWxCClose,
    FWxCGetPositionCount
)

def list_devices():
    # Retrieves and returns a list of available filter wheel device IDs.
    devices = FWxCListDevices()
    try:
        if isinstance(devices, bytes):
            txt = devices.decode('utf-8').strip()
            dev_list = [d for d in txt.replace(',', ' ').split() if d]
        else:
            dev_list = [dev[0] for dev in devices]
    except Exception:
        dev_list = [dev[0] for dev in devices]
    return dev_list



def open_device(device_id, nBaud=115200, timeout=3):
    if isinstance(device_id, bytes):
        serial_str = device_id.decode('utf-8')
    else:
        serial_str = device_id

    handle = FWxCOpen(serial_str, nBaud, timeout)
    if handle < 0:
        raise RuntimeError(f"Failed to open device {serial_str} (error code {handle})")
    return handle


def get_position(handle):
    # Gets the current position of the filter wheel.
    pos = [0]
    result = FWxCGetPosition(handle, pos)
    if result < 0:
        raise RuntimeError(f"GetPosition failed with error code {result}")
    return pos[0]

def get_position_count(handle):
    # Returns the total number of positions on the wheel.
    count = [0]
    result = FWxCGetPositionCount(handle, count)
    if result < 0:
        raise RuntimeError(f"GetPositionCount failed (error code {result})")
    return count[0]


def set_position(handle, position):
    # Sets the filter wheel to the given position (integer index).
    result = FWxCSetPosition(handle, int(position))
    if result < 0:
        raise RuntimeError(f"SetPosition({position}) failed with error code {result}")


def close_device(handle):
    # Closes the connection to the filter wheel.
    FWxCClose(handle)

# loop for controlling filter wheels directly
def main():
    print("Finding filter wheel devices...")
    devices = list_devices()
    if not devices:
        print("No filter wheel devices found.")
        sys.exit(1)

    print("Found devices:")
    for idx, dev in enumerate(devices):
        print(f" [{idx}] {dev}")

    while True:
        selected = 0
        if len(devices) > 1:
            choice = input(f"Select device index [0-{len(devices)-1}], or 'exit': ")
            if choice.strip().isdigit():
                selected = int(choice)
            elif choice.strip().lower() == "exit":
                break
        device_id = devices[selected]

        print(f"Opening device '{device_id}' with baud=115200, timeout=3s...")
        handle = open_device(device_id)
        print("Device opened.")

        max_pos = get_position_count(handle)
        # print(f"total positions: {max_pos}")

        try:
            while True:
                cmd = input("(get/set/exit) > ").strip().lower()
                if cmd == 'get':
                    pos = get_position(handle)
                    print(f"Current position: {pos}")
                elif cmd.startswith('set'):
                    parts = cmd.split()
                    if len(parts) != 2 or not parts[1].isdigit():
                        print(f"Usage: set <position_index (0-{max_pos})>")
                    else:
                        if int(parts[1])>max_pos:
                            print(f"Usage: set <position_index (0-{max_pos})>")
                        else:
                            set_position(handle, int(parts[1]))
                            print(f"Set position to {parts[1]}")
                elif cmd == 'exit':
                    break
                else:
                    print("Unknown command. Use 'get', 'set <n>', or 'exit'.")
        finally:
            print("Closing device...")
            close_device(handle)
            print("Device closed")


if __name__ == '__main__':
    main()