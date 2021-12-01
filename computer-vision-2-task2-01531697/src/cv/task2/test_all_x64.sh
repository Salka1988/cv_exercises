#!/bin/bash

# compare/evaluation script for cv/task2 on 64
# this script is build on imagemagick

# use "sudo apt-get install imagemagick" to install imagemagick
# use "./test_all_x64.sh" to run the script
# use "chmod 755 test_all_x64.sh" for "Permission denied" error

echo ""
echo "imagemagick compare script for cv/task2 on x64" 
echo "- comparing output with reference images"
echo "- creates difference images in 'dif/'"
echo "- a 'correct' image will be completely white"
echo "- read comments in this file for more info"
echo ""

rm -r ./dif
mkdir -p dif/


# declaration of all testcases
declare -a arr=(`ls ./data/ref_x64/`)

#every loop is one testcase
for i in "${arr[@]}"
do
  echo 'Testing' ${i} '...'
  mkdir -p dif/${i}
  mkdir -p dif/${i}/normal
  mkdir -p dif/${i}/bonus

  # declaration of all images
  declare -a img_norm=(`ls ./data/ref_x64/${i}/normal`)
  #every loop is one testcase
  for img in "${img_norm[@]}"
  do
    convert data/ref_x64/${i}/normal/${img} output/${i}/normal/${img} -compose difference -composite -negate -contrast-stretch 0 dif/${i}/normal/${img}
  done

  # declaration of all images
  declare -a img_bon=(`ls ./data/ref_x64/${i}/bonus`)
  #every loop is one testcase
  for img in "${img_bon[@]}"
  do
    convert data/ref_x64/${i}/bonus/${img} output/${i}/bonus/${img} -compose difference -composite -negate -contrast-stretch 0 dif/${i}/bonus/${img}
  done
  echo ${i}' done.'
done


