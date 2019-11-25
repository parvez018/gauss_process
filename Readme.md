## Gaussian Process in Python
I learned about Gaussian process a month ago. Gaussian processes are used to estimate a blackbox function, a function for which nothing is known except a few samples. Under a few assumption, they can give out pretty good estimate of the function value at uknown points. This is my attempt at implementing Gaussian process from scratch in python. 

The code is mostly taken from the [this blogpost](https://katbailey.github.io/post/gaussian-processes-for-dummies/). This blog has a very good introduction to Gaussian process for beginners. The code provided in the blog implements assumes a zero prior function, while my implementation contains **both zero and non-zero prior function**.
