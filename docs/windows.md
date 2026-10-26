# Windows 本地工作目录

项目已可在本地 E: 盘直接开发。目标路径：

~~~text
E:\2_项目代码\single spark can start a prairie fire
~~~

## 目录尚未初始化为仓库

在 PowerShell 中执行；`git clone` 要求目标目录不存在或为空，已有文件应先自行确认：

~~~powershell
git clone https://github.com/never-die-cold/single-spark-can-start-a-prairie-fire.git "E:\2_项目代码\single spark can start a prairie fire"
Set-Location "E:\2_项目代码\single spark can start a prairie fire"
~~~

## 目录已经是该远程仓库的工作副本

~~~powershell
Set-Location "E:\2_项目代码\single spark can start a prairie fire"
git status
git remote -v
git pull --ff-only
~~~

保留本地未提交修改；不要为了同步直接执行 `reset --hard` 或覆盖整个目录。

## 安装与验证

使用 Python 3.10+。下面直接调用虚拟环境解释器，不需要更改 PowerShell 脚本执行策略：

~~~powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m prairie_fire.experiments --output results/latest.json
.\.venv\Scripts\python.exe -m prairie_fire.training --output results/m1-latest.json
~~~

快速运行：

~~~powershell
.\.venv\Scripts\python.exe -m prairie_fire.experiments --quick
~~~

打开 [m1-training.md](m1-training.md) 查看完整小网络训练及结果解释，按 [roadmap.md](roadmap.md) 继续 M1.1。当前结果只有单进程模拟；仓库中没有真实设备 Agent 或分布式服务需要启动。
