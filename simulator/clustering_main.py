import subprocess

# 명령어 리스트
commands = [
    'python3 ./tensorflow_cluster.py',
    'python3 ./pytorch_cluster.py',
    'python3 ./graphlearn_cluster.py'
]

# 명령어 실행
for command in commands:
    try:
        subprocess.run(command, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while executing {command}: {e}")