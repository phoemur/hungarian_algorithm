# hungarian_algorithm
[![Build Status](https://travis-ci.org/phoemur/hungarian_algorithm.svg?branch=master)](https://travis-ci.org/phoemur/hungarian_algorithm)


This is an implementation of the Hungarian algorithm in C++
The Hungarian algorithm, also know as Munkres or Kuhn-Munkres
algorithm is usefull for solving the assignment problem.

This implementation is uses the matrix-based solution, instead
of bipartite-graphs matching.
 
Assignment problem: Let C be an n x n matrix 
representing the costs of each of n workers to perform any of n jobs.
The assignment problem is to assign jobs to workers so as to 
minimize the total cost. Since each worker can perform only one job and 
each job can be assigned to only one worker the assignments constitute 
an independent set of the matrix C.
 
It is a port heavily based on http://csclab.murraystate.edu/~bob.pilgrim/445/munkres.html
 
This version is written by Fernando B. Giannasi - feb/2018
