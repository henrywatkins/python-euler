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

#### Problem 11

iterate through matrix elements with double for loop, create horizontal, vertical, right diagonal and left diagonal groups of 4 for each index, then add to a product list. return the max valued product.

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

#### Problem 18

Each of the paths through the triangle can be represented by a binary number. There are 2^N total paths where N is the depth of the triangle. AS you loop through the enumerated paths, find the binary number, then the column index for that row in the path is the sum of all the binary digits up to that row. Loop through all paths to find max value.

#### Problem 20

calculate factorial, convert to string, then sum digits

#### Problem 21

Use prime factorization then sum of divisors using
$$
\sigma(n) = \prod_{i=1}^k \frac{p_i^{e_i + 1} -1 }{p_i -1}
$$
find all the amicable numbers by simple search of all below max number.

#### Problem 22

load in file with `with open` then trip and split the names into a list. Sort with `sorted`, then use `enumerate` to get index and multiply with name score. This names score comes from `dict` with enumerated alphabet values.

#### Problem 24

Create the lexicographic permutations of the numbers 0 to 9. There is a standard djikstra's algorithm for this. Find the rightmost index that is larger than its left neighbour, then swap all the elements up to the next value. Then 

#### Problem 25

Find first fibonacci number with specific number of digits.

Use dummy variables to count up the fibonacci numbers in a while loop up until the len(str(fn)) reaches right length.

#### Problem 34

iterate through numbers, use `str` and `math` package `factorial` to check if sum of factorials of digits equals number. Just chose a large number to stop at (100000).

#### Problem 35

Generate list of primes with eratosthenes, then rotate the primes by shifting the string version with array slicing. Check each rotation is prime by getting its prime factors. Add to circular prime list if all rotations are prime.

#### Problem 36

Use the bin function to find binary version of number, then use reversed to check if palindrome; add to list and sum.

#### Problem 39

Like problem 9 , find pythagorean triplet solutions using euclid formula. remember to remove duplicate solutions, then search for the best in range 1<=p<=1000

#### Problem 40

create champerdowne constant by string concatenation, then just find the product of digits and index locations.

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

#### Problem 63

Use fact that the number of digits in decimal number can be found with log
$$
N_d = \lfloor\log_{10}(X)\rfloor +1
$$
Problem is then to find the number of solutions (d, x) that satisfy
$$
d-1 = \lfloor d\log_{10}(x)\rfloor
$$
For a specific value of d, that largest value x can take is 10, and so loop over values of d and x = 1-10 and count those that satisfy the above expression.