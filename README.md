# Udacity - Deep Learning Foundation

This repository contains the demo codes and assignments I have done during the [Udacity Deep Learning Foundation][udacity-deep-learning-foundation].

**Disclaimer:** there is probably a code of conduct if you are enrolled in the nanodegree. Copying the code from the repository and pretending it to be sure would be bad. Even looking into it may be an infringement. Talk to your instructor.

The course is just great! You will use and develop neural networks for image recognition with convolution, natural language processing with embeddings and character based text generation with RNN/LTSM.

## Installation and setup

Since the datasets are loaded in memory, 4 GB of RAM are short to run the notebooks, 8 GB will be more comfortable. The instructions are given for Mac OS, but it works on Linux as well.

I recommend to install the Python bundle [Anaconda][anaconda]. All the assignments have been done with Python 3, so download the installation script accordingly. You can then install [Jupyter Notebook][jupyter] from ``conda``.

The next step is to install TensorFlow. At the time of writing, the latest release is r1.0, after several upgrades during the course. You can refer to the [official documentation][tensorflow-anaconda-install]. Here is the list of commands:

```
$ conda create -n tensorflow python=3
$ source activate tensorflow
```

You can now grab the assignments from here to run my code or the stubs from the [TensorFlow repository][tensorflow-repo]. All the other prerequisites (like ``numpy``) have to be installed on the fly.

## Quick start

From the repository root, start the Notebook server with ``jupyter notebook``.

## Credits

All the assignments come from [Udacity][udacity-deep-learning-foundation] and the [TensorFlow repository][tensorflow-repo]

[udacity-deep-learning-foundation]: https://www.udacity.com/course/deep-learning-nanodegree-foundation--nd101
[udacity-nanodegree]: https://www.udacity.com/nanodegree

[jupyter]: http://jupyter.org/
[anaconda]: https://www.continuum.io/
[tensorflow-repo]: https://github.com/tensorflow/tensorflow/tree/master/tensorflow/examples/udacity
