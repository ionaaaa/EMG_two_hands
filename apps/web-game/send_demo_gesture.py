import argparse
import json
import urllib.request


def main() -> None:
    parser = argparse.ArgumentParser(description="Send a demo gesture to the EMG bridge.")
    parser.add_argument("gesture", nargs="?", default="fist", help="rest / fist / open-palm / pinch")
    parser.add_argument("confidence", nargs="?", type=float, default=0.93)
    parser.add_argument("--hand", default=None, help="left / right (l / r). 省略则不区分。")
    args = parser.parse_args()

    payload = {"gesture": args.gesture, "confidence": args.confidence}
    if args.hand:
        payload["hand"] = args.hand

    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        "http://127.0.0.1:8765/gesture",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with urllib.request.urlopen(request, timeout=5) as response:
        print(response.read().decode("utf-8"))


if __name__ == "__main__":
    main()
