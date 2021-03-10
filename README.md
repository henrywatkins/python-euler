# Euler Problems

This repository contains solutions to problems from the Project Euler website www.projecteuler.net. 

Solution checking is performed via pytest unit tests. 

## Tricks/Methods

#### Problem 12

Use a prime factorization method (trial division up to square root). Then use fact that number of divisors satisfies
$$
d(n) = \prod_{i=1}^k (e_k+1)
$$
where e_k are the powers of the prime factor representation of n. 

#### Problem 16

dynamic typing makes this easy, just get 2**1000 then convert to string then to individual digits and sum