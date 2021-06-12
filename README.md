# Euler Problems

This repository contains solutions to problems from the Project Euler website www.projecteuler.net. 

Solution checking is performed via pytest unit tests. 

## Tricks/Methods

#### Problem 5

Given a set of necessary divisors, find the smallest number with that set among its divisors. The prime factorization of each divisor will be a subset of the prime factorisation of the number. Find the prime factorisations of the divisors, then find the max power of each prime necessary to create each divisor. These powers are the used to find the final answer
$$
n = \prod_i^kp_i^{e_i}
$$

#### Problem 8

Use a moving window across the list of digits, find the max.

#### Problem 9

use euclid formula to get integer solutions to pythagoras eq, under the constraint of a+b+c=p. 

#### Problem 12

Use a prime factorization method (trial division up to square root). Then use fact that number of divisors satisfies
$$
d(n) = \prod_{i=1}^k (e_i+1)
$$
where e_k are the powers of the prime factor representation of n. 

#### Problem 15

This is a graph path counting problem. Create the adjacency matrix of the graph (directed graph so not symmetric, no loops). The number of paths of length k between vertex i and j is given by the power of the adjacency matrix.
$$
C_k = A^k
$$


#### Problem 16

dynamic typing makes this easy, just get 2**1000 then convert to string then to individual digits and sum

#### Problem 20

calculate factorial, convert to string, then sum digits

#### Problem 21

Use prime factorization then sum of divisors using
$$
\sigma(n) = \prod_{i=1}^k \frac{p_i^{e_i + 1} -1 }{p_i -1}
$$
find all the amicable numbers by simple search of all below max number.

#### Problem 25

Find first fibonacci number with specific number of digits.

Use dummy variables to count up the fibonacci numbers in a while loop up until the len(str(fn)) reaches right length.

#### Problem 39

Like problem 9 , find pythagorean triplet solutions using euclid formula. remember to remove duplicate solutions, then search for the best in range 1<=p<=1000

#### Problem 47

create a list of consecutive numbers, find their prime factors then check the length of the set{}, if all prime factor sets the right length, return the result

#### Problem 48

Use binary exponentiation to find large self powers, then sum up 

#### Problem 52

while loop iterating up through the positive numbers, look at a sorted list of digits, and check if all the sorted digit lists of all multiples are equal

#### Problem 56

Use binary exponentiation to find large powers, convert to strings then sum digits. Find the largest

#### Problem 57

Use recurrence relations for the convergents of an infinite continued fraction.
$$
h_n = a_n h_{n-1} + h_{n-2}\\
k_n = a_n k_{n-1} + h_{n-2}
$$
For the numerator and denominator of the nth rational approximation respectively.