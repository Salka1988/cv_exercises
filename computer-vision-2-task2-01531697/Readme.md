# Overview 

This repository contains the assignments for "Computer Vision 2" - 710.006 for SS 2019.

## Obtaining the Code

To obtain the code, you first have to do the following things:

* Login into https://assignments.icg.tugraz.at (use the TUG Online Button)
* Go to your profile and enter your public SSH key(s)

You then just have to open a terminal and type:

    git clone <clone-url-of-your-repo>

where you can obtain the <clone-url-of-your-repo> from the blue ``Clone'' button in your repository.

To upload your code to the repository, just type:

    git push origin master

To make a submission, you have to checkout a ``submission'' branch and push this branch to the server and type:

    git checkout -b submission
    git push origin submission


## Requirements

We build and run the code on an Ubuntu 16.04 machine with the default OpenCV (2.4.9) and g++ version (5.4).
To test your code in this environment you simply need to push your code into the git repository (git push origin master or 
git push origin master).

Every push will be compiled and tested in our test system. We also store the output and generated images of your program.
As your code will be graded in this testing environment, we strongly recommend that you check that your program compiles 
and runs as you intended (i.e. produces the same output images).
To view the output and generated images of your program, just click on the CI/CD tab -> ``Pipelines''. For every commit,
which you push into the repository, a test run is created. The images of the test runs
are stored as artifacts. Just click on the test run and then you should see the output of your program. On the right 
side of your screen there should be an artifacts link.

We also provide a virtual box image with a pre-installed Ubuntu 16.04: https://cloud.tugraz.at/index.php/s/ZzZorSHsTbp92Ki

In the Teach Center we also provide a detailed description how you can build your own virtual box image.

## Compiling the Code

We use cmake to build our framework. If you are with a linux shell at the root of your repository, just type:

    repopath $ cd src/
    repopath/src $ mkdir build
    repopath/src $ cd build
    repopath/src/build $ cmake ../
    repopath/src/build $ make


To run your program (task2), just type:

    repopath/src/build $ cd ../cv/task2/
    repopath/src/cv/task1a/ $ ../../build/cv/task2/cvtask2 <testcase.json>

## Making a Submission

As mentioned in the previous section, to make a submission, you need to create the ``submission'' branch and 
push it into your private repository, which you created via courseware. If you cloned your repository from 
your private repository, you just need to type:

    repopath $ git checkout -b submission
    repopath $ git push origin submission

No you will see the submission branch in your gitlab webinterface.
