# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.10

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/salka1988/Documents/CV2/computer-vision-2-task2-01531697/src

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/salka1988/Documents/CV2/computer-vision-2-task2-01531697/src/build

# Include any dependencies generated for this target.
include cv/task2/CMakeFiles/cvtask2.dir/depend.make

# Include the progress variables for this target.
include cv/task2/CMakeFiles/cvtask2.dir/progress.make

# Include the compile flags for this target's objects.
include cv/task2/CMakeFiles/cvtask2.dir/flags.make

cv/task2/CMakeFiles/cvtask2.dir/main.cpp.o: cv/task2/CMakeFiles/cvtask2.dir/flags.make
cv/task2/CMakeFiles/cvtask2.dir/main.cpp.o: ../cv/task2/main.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/salka1988/Documents/CV2/computer-vision-2-task2-01531697/src/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object cv/task2/CMakeFiles/cvtask2.dir/main.cpp.o"
	cd /home/salka1988/Documents/CV2/computer-vision-2-task2-01531697/src/build/cv/task2 && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/cvtask2.dir/main.cpp.o -c /home/salka1988/Documents/CV2/computer-vision-2-task2-01531697/src/cv/task2/main.cpp

cv/task2/CMakeFiles/cvtask2.dir/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/cvtask2.dir/main.cpp.i"
	cd /home/salka1988/Documents/CV2/computer-vision-2-task2-01531697/src/build/cv/task2 && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/salka1988/Documents/CV2/computer-vision-2-task2-01531697/src/cv/task2/main.cpp > CMakeFiles/cvtask2.dir/main.cpp.i

cv/task2/CMakeFiles/cvtask2.dir/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/cvtask2.dir/main.cpp.s"
	cd /home/salka1988/Documents/CV2/computer-vision-2-task2-01531697/src/build/cv/task2 && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/salka1988/Documents/CV2/computer-vision-2-task2-01531697/src/cv/task2/main.cpp -o CMakeFiles/cvtask2.dir/main.cpp.s

cv/task2/CMakeFiles/cvtask2.dir/main.cpp.o.requires:

.PHONY : cv/task2/CMakeFiles/cvtask2.dir/main.cpp.o.requires

cv/task2/CMakeFiles/cvtask2.dir/main.cpp.o.provides: cv/task2/CMakeFiles/cvtask2.dir/main.cpp.o.requires
	$(MAKE) -f cv/task2/CMakeFiles/cvtask2.dir/build.make cv/task2/CMakeFiles/cvtask2.dir/main.cpp.o.provides.build
.PHONY : cv/task2/CMakeFiles/cvtask2.dir/main.cpp.o.provides

cv/task2/CMakeFiles/cvtask2.dir/main.cpp.o.provides.build: cv/task2/CMakeFiles/cvtask2.dir/main.cpp.o


# Object files for target cvtask2
cvtask2_OBJECTS = \
"CMakeFiles/cvtask2.dir/main.cpp.o"

# External object files for target cvtask2
cvtask2_EXTERNAL_OBJECTS =

cv/task2/cvtask2: cv/task2/CMakeFiles/cvtask2.dir/main.cpp.o
cv/task2/cvtask2: cv/task2/CMakeFiles/cvtask2.dir/build.make
cv/task2/cvtask2: /usr/local/lib/libopencv_dnn.so.4.1.0
cv/task2/cvtask2: /usr/local/lib/libopencv_gapi.so.4.1.0
cv/task2/cvtask2: /usr/local/lib/libopencv_ml.so.4.1.0
cv/task2/cvtask2: /usr/local/lib/libopencv_objdetect.so.4.1.0
cv/task2/cvtask2: /usr/local/lib/libopencv_photo.so.4.1.0
cv/task2/cvtask2: /usr/local/lib/libopencv_stitching.so.4.1.0
cv/task2/cvtask2: /usr/local/lib/libopencv_video.so.4.1.0
cv/task2/cvtask2: /usr/local/lib/libopencv_calib3d.so.4.1.0
cv/task2/cvtask2: /usr/local/lib/libopencv_features2d.so.4.1.0
cv/task2/cvtask2: /usr/local/lib/libopencv_flann.so.4.1.0
cv/task2/cvtask2: /usr/local/lib/libopencv_highgui.so.4.1.0
cv/task2/cvtask2: /usr/local/lib/libopencv_videoio.so.4.1.0
cv/task2/cvtask2: /usr/local/lib/libopencv_imgcodecs.so.4.1.0
cv/task2/cvtask2: /usr/local/lib/libopencv_imgproc.so.4.1.0
cv/task2/cvtask2: /usr/local/lib/libopencv_core.so.4.1.0
cv/task2/cvtask2: cv/task2/CMakeFiles/cvtask2.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/salka1988/Documents/CV2/computer-vision-2-task2-01531697/src/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable cvtask2"
	cd /home/salka1988/Documents/CV2/computer-vision-2-task2-01531697/src/build/cv/task2 && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/cvtask2.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
cv/task2/CMakeFiles/cvtask2.dir/build: cv/task2/cvtask2

.PHONY : cv/task2/CMakeFiles/cvtask2.dir/build

cv/task2/CMakeFiles/cvtask2.dir/requires: cv/task2/CMakeFiles/cvtask2.dir/main.cpp.o.requires

.PHONY : cv/task2/CMakeFiles/cvtask2.dir/requires

cv/task2/CMakeFiles/cvtask2.dir/clean:
	cd /home/salka1988/Documents/CV2/computer-vision-2-task2-01531697/src/build/cv/task2 && $(CMAKE_COMMAND) -P CMakeFiles/cvtask2.dir/cmake_clean.cmake
.PHONY : cv/task2/CMakeFiles/cvtask2.dir/clean

cv/task2/CMakeFiles/cvtask2.dir/depend:
	cd /home/salka1988/Documents/CV2/computer-vision-2-task2-01531697/src/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/salka1988/Documents/CV2/computer-vision-2-task2-01531697/src /home/salka1988/Documents/CV2/computer-vision-2-task2-01531697/src/cv/task2 /home/salka1988/Documents/CV2/computer-vision-2-task2-01531697/src/build /home/salka1988/Documents/CV2/computer-vision-2-task2-01531697/src/build/cv/task2 /home/salka1988/Documents/CV2/computer-vision-2-task2-01531697/src/build/cv/task2/CMakeFiles/cvtask2.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : cv/task2/CMakeFiles/cvtask2.dir/depend
