import subprocess
import sys

def install_ffmpeg():
    print("Starting FFmpeg installation...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "setuptools"])
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "ffmpeg-python"])
        print("FFmpeg installation successful.")
    except subprocess.CalledProcessError:
        print("FFmpeg installation failed. Please check your internet connection and try again.")
        sys.exit(1)
        
    try:
        subprocess.check_call([
            "wget",
            "https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz",
            "-O",
            "/tmp/ffmpeg.tar.xz"
        ])
        subprocess.check_call([
            "tar",
            "-xf",
            "/tmp/ffmpeg.tar.xz",
            "-C",
            "/tmp"
        ])

        results = subprocess.run([
            "find", "/tmp", "-name", "ffmpeg", "-type", "f" 
        ], capture_output=True, text=True)
        ffmpeg_path = results.stdout.strip()

        subprocess.check_call(["cp", ffmpeg_path, "/usr/local/bin/ffmpeg"])

        subprocess.check_call(["chmod", "+x", "/usr/local/bin/ffmpeg"])
        print("Installed static FFmpeg binary successfully.")
    except Exception as e:
        print(f"Failed to install static FFmpeg binary: {e}")


    try:
        result = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True)
        print(result.stdout)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("FFmpeg installation verification failed. Please check your installation.")
        return False

        
