cd src
cd build
make clean
cmake ../
make
cd ../cv/task1b
echo "Building coins1 ..."
../../build/cv/task1b/cvtask1b 0
echo "DONE"
echo "Building coins3 ..."
../../build/cv/task1b/cvtask1b 1
echo "DONE"
echo "Building coins_test_3 ..."
../../build/cv/task1b/cvtask1b 2
echo "DONE"
echo "Generating dif files ..."
bash test_all_x64.sh
echo "DONE"
