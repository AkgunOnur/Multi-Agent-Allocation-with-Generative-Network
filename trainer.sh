#! /bin/bash


# this code does training and testing using parameters from json files in hyperparameters folder


#######################################################################################

# train with same map size and curriculum mode for now

typ="no curriculum"
map_lim="map = 10x10"
model="PPO"
reward="negative reward = -0.5"

number_of_files=$(ls -1q hyperparameters | wc -l)
printf "${number_of_files} json files will be trained\n"
#sleep 1

# !!! enter new models index
index=4
FILE="seko_models/model${index}"

if [[ -d "$FILE" ]]
then
    echo "$FILE exists on your filesystem."
    exit 1
fi


for json_file in $(ls hyperparameters)
do
    # cpu training is louder, cuda is pretty silent
    # training speeds are close
    python3 train.py --hyperparameters="hyperparameters/${json_file}" --cuda=1 --train_steps=10000
    sleep 1

    # mode:={0 => nocur, 1 => cur}
    python3 test.py --mode=0

    FILE="seko_models/model${index}"
    mkdir "${FILE}"
    touch info.txt


    echo "$typ" >> info.txt
    echo "$model" >> info.txt
    echo $(jq -r '.policy_kwargs' "hyperparameters/${json_file}") >> info.txt
    echo "$reward" >> info.txt
    echo "$map_lim" >> info.txt

    # moves cur/nocur with any map size
    mv output_*cur*   "${FILE}"
    mv info.txt       "${FILE}"

    mv iterations_*cur*.pickle "${FILE}"
    mv results_*cur*.pickle    "${FILE}"

    printf "\n\ndone ${index}th step\n\n"
    index=$index+1
    sleep 2
done


#eof
