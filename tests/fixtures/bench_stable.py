from __future__ import annotations

import time


def main() -> None:
    checksum = 0
    for round_id in range(4):
        factor = round_id + 1
        for i in range(90_000):
            checksum = (checksum + (((i * factor) % 97) * ((i + factor) % 89))) % 1_000_000_007
        time.sleep(0.01)

    print(f"checksum={checksum}")


if __name__ == "__main__":
    main()
