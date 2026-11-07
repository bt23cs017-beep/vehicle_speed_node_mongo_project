import argparse
import subprocess
import sys
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Count vehicles and estimate calibrated speed.")
    parser.add_argument("input_video", help="Input video path")
    parser.add_argument("output_video", nargs="?", help="Optional output video path")
    arguments = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    processor = Path(__file__).with_name("tracking_test.py")
    input_video = Path(arguments.input_video)
    if not input_video.is_absolute():
        input_video = project_root / input_video

    command = [sys.executable, str(processor), str(input_video)]
    if arguments.output_video:
        output_video = Path(arguments.output_video)
        if not output_video.is_absolute():
            output_video = project_root / output_video
        command.append(str(output_video))

    return subprocess.run(command, cwd=project_root, check=False).returncode

if __name__ == "__main__":
    raise SystemExit(main())
