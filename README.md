# Overview

The goal of the project is to implement a 2D Particle Filter for estimating the
position of a vehicle using the noisy sensor measurements. 

This project should be run in the Simulator which can be downloaded [here](https://github.com/udacity/self-driving-car-sim/releases)

Cmake is used as a build system for the project. In order to compile the code:
1. `mkdir build`
2. `cd build`
3. `cmake ../src/`
4. `make`
5. `./particle_filter`

## Important Dependencies

The project has been tested / run on Ubuntu.

* cmake >= 3.5
* make >= 4.1 (Linux, Mac),
* gcc/g++ >= 5.4


## Particle filter bird's-eye-view:

* particle initialization around the measurement;
* predicting particle movement using CTRV model;
* associating sensor measurements with map landmarks;
* updating particle weights based on observations;
* re-sampling with replacement depending the particle weights.

