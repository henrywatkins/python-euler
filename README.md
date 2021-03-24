# Euler Problems

This repository contains solutions to problems from the Project Euler website www.projecteuler.net. 

Solution checking is performed via pytest unit tests. 

## Tricks/Methods

#### Problem 12

Use a prime factorization method (trial division up to square root). Then use fact that number of divisors satisfies
$$
d(n) = \prod_{i=1}^k (e_i+1)
$$
where e_k are the powers of the prime factor representation of n. 

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

#### Problem 48

Use binary exponentiation to find large self powers, then sum up 