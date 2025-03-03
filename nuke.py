"""
Kills every
- `/lcc/.venv/bin/python3` process listed by `nvidia-smi`,
- every `/lcc/.venv/bin/python3 -m lcc` process listed by `ps`.
- every `pt_main_thread` process listed by `ps`.
"""

import re
import subprocess
from getpass import getuser
from time import sleep

ART = """

        ⣀⣠⣀⣀  ⣀⣤⣤⣄⡀
   ⣀⣠⣤⣤⣾⣿⣿⣿⣿⣷⣾⣿⣿⣿⣿⣿⣶⣿⣿⣿⣶⣤⡀
 ⢠⣾⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣷
 ⢸⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣶⡀
 ⢀⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡇
 ⢸⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡿⢿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠿⠟⠁
  ⠻⢿⡿⢿⣿⣿⣿⣿⠟⠛⠛⠋⣀⣀⠙⠻⠿⠿⠋⠻⢿⣿⣿⠟
      ⠈⠉⣉⣠⣴⣷⣶⣿⣿⣿⣿⣶⣶⣶⣾⣶
        ⠉⠛⠋⠈⠛⠿⠟⠉⠻⠿⠋⠉⠛⠁
              ⣶⣷⡆
      ⢀⣀⣠⣤⣤⣤⣤⣶⣿⣿⣷⣦⣤⣤⣤⣤⣀⣀
    ⢰⣿⠛⠉⠉⠁   ⢸⣿⣿⣧    ⠉⠉⠙⢻⣷
     ⠙⠻⠷⠶⣶⣤⣤⣤⣿⣿⣿⣿⣦⣤⣤⣴⡶⠶⠟⠛⠁
          ⢀⣴⣿⣿⣿⣿⣿⣿⣷⣄
         ⠒⠛⠛⠛⠛⠛⠛⠛⠛⠛⠛⠓

"""

if __name__ == "__main__":
    print(ART)

    raw = subprocess.check_output(["nvidia-smi"])
    pids = []
    r = re.compile(
        r"\|\s+\d\s+N/A\s+N/A\s+(\d+)\s+C.*benchmark/.venv/bin/python3"
    )
    for line in raw.decode("utf-8").split("\n"):
        if m := re.search(r, line):
            pids.append(m.group(1))
    if not pids:
        print("[nvidia-smi] No matching processes found.")
    else:
        print("[nvidia-smi] Nuking processes:", " ".join(pids))
        try:
            subprocess.run(["kill", "-9"] + pids, check=True)
        except Exception as e:
            print("Nuke didn't detonate:", e)

    print("Sleeping for 1 second")
    sleep(1)

    USER = getuser()
    raw = subprocess.check_output(["ps", "-u", USER, "-eo", "pid,comm"])
    pids = []
    r = re.compile(r"(\d+) .*/lcc/\.venv/bin/python3.*m lcc.*")
    for line in raw.decode("utf-8").split("\n"):
        if m := re.search(r, line):
            pids.append(m.group(1))
    if not pids:
        print("[ps python3] No matching processes found.")
    else:
        print("[ps python3] Nuking processes:", " ".join(pids))
        try:
            subprocess.run(["kill", "-9"] + pids, check=True)
        except Exception as e:
            print("Nuke didn't detonate:", e)

    print("Sleeping for 1 second")
    sleep(1)

    raw = subprocess.check_output(["ps", "-u", USER, "-eo", "pid,comm"])
    pids = []
    r = re.compile(r"(\d+) pt_main_thread")
    for line in raw.decode("utf-8").split("\n"):
        if m := re.search(r, line):
            pids.append(m.group(1))
    if not pids:
        print("[ps pt_main_thread] No matching processes found.")
    else:
        print("[ps pt_main_thread] Nuking processes:", " ".join(pids))
        try:
            subprocess.run(["kill", "-9"] + pids, check=True)
        except Exception as e:
            print("Nuke didn't detonate:", e)
