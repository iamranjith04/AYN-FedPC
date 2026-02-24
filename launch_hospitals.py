import subprocess
import sys
import time

PYTHON = sys.executable
print("debug")
hospitals = ["hospital_1", "hospital_2", "hospital_3"]

for h in hospitals:
    subprocess.Popen(
        [PYTHON, "-m", "hospital.client", h],
        creationflags=subprocess.CREATE_NEW_CONSOLE
    )
    time.sleep(1)

print("✅ All hospitals launched")
