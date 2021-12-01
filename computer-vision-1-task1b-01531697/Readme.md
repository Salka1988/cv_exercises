# Overview 

This repository contains the assignments for "Computer Vision 1" - 710.005 for SS 2019.

## Obtaining the Code

To obtain the code, you first have to do the following things:

* Login into https://assignments.icg.tugraz.at (use the TUG Online Button)
* Go to your profile and enter your public SSH key(s)

If you have already created your own repository with courseware, you just have to open a terminal and type:

    git clone <clone-url-of-your-repo>

where you can obtain the <clone-url-of-your-repo> from the blue ``Clone'' button in your repository.

If you want clone this repository from https://assignments.icg.tugraz.at/CV/cv1_2019 with the public HTTP link, type:

    git clone https://assignments.icg.tugraz.at/CV/cv1_2019.git

Later, if you want to make your submission, you have to create a repository via courseware and add it as remote repo.
From your private repository, obtain the <clone-url-of-your-repo> vie the blue ``Clone'' button.
Then, add your private repository to your local git repository via:

    git remote add myrepo <clone-url-of-your-repo>

To upload your code to the repository, just type:

    git push myrepo master

To make a submission, you have to checkout a ``submission'' branch and push this branch to the server and type:

    git checkout -b submission
    git push myrepo submission


## Requirements

We build and run the code on an Ubuntu 16.04 machine with the default OpenCV (2.4.9) and g++ version (5.4).
To test your code in this environment you simply need to push your code into the git repository (git push origin master or 
git push myrepo master).

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


To run your program (task1a), just type:

    repopath/src/build $ cd ../cv/task1a/
    repopath/src/cv/task1a/ $ ../../build/cv/task1a/cvtask1a <testcase.json>

If you want to run task1b, just type:

    repopath/src/build $ cd ../cv/task1b/
    repopath/src/cv/task1b/ $ ../../build/cv/task1b/cvtask1b 0

or:

    repopath/src/build $ cd ../cv/task1b/
    repopath/src/cv/task1b/ $ ../../build/cv/task1b/cvtask1b 1

The argument for task1b is just the index of the test-case. As there is just three testcases you might pass 0, 1 or 2 as argument
for the program.

## Making a Submission

As mentioned in the previous section, to make a submission, you need to create the ``submission'' branch and 
push it into your private repository, which you created via courseware. If you cloned your repository from 
your private repository, you just need to type:

    repopath $ git checkout -b submission
    repopath $ git push origin submission

If you cloned your repository from the public repository (https://assignments.icg.tugraz.at/CV/cv1_2019 ),
you first need to add your private repository as remote. To do this, you need to uptain the <clone-url-of-your-repo> link
via the blue ``Clone'' button in your repository. Then, type:

    repopath $ git remote add myrepo <clone-url-of-your-repo>
    repopath $ git push myrepo submission

No you will see the submission branch in your gitlab webinterface.
