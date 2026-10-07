@echo off
cd /d "%~dp0sumo"
netconvert -n nodes.nod.xml -e edges.edg.xml -o network.net.xml
pause
