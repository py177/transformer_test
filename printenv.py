import subprocess

# 运行 pip freeze 并获取所有已安装的包
pip_freeze_output = subprocess.run(["pip", "freeze"], capture_output=True, text=True).stdout

# 保存到 requirements.txt
with open("requirements.txt", "w", encoding="utf-8") as f:
    f.write(pip_freeze_output)

print("环境依赖已保存到 requirements.txt，可用 pip install -r requirements.txt 重新安装！")
