#!/bin/bash
for i in `seq $1 $2`;
do
	echo -e "\n\n ******** started trial $i ******** \n\n"
    time C:\\Users\\avivra\\PycharmProjects\\pythonProject\\venv\\Scripts\\python.exe main.py --config-name=config$i hydra.run.dir=\"outputs/$3/$i\"
	#echo C:\\Users\\avivra\\PycharmProjects\\pythonProject\\venv\\Scripts\\python.exe main.py --config-name=config$i hydra.run.dir=\"outputs/$3/\${now:%H-%M-%S}\"
	echo -e "\n\n ******** finished trial $i ******** \n\n"

done