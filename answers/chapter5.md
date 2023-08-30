# 5.1.1 Vectors

1. Dot product
    1. [E] What’s the geometric interpretation of the dot product of two vectors?\
       [A] In the geometric context, the dot product of 2 vecotrs is projection of one vector over another.
        For the given geometric vectors $\overrightarrow A$ and $\overrightarrow B$, their dot product is given by\
           $$\overrightarrow A.\overrightarrow B = |\overrightarrow A|.|\overrightarrow B|.cos\theta$$
       
       ![vectors](./vectors1.png)
       
    3. [E] Given a vector $u$, find vector $v$ of unit length such that the dot product of $u$ and $v$ is maximum.\
       [A] Dot product of $\vec{u}$ and $\vec{v}$ is givenby $|\vec{u}|.|\vec{v}|.cos\theta$. Where $|\vec{u}|$ and $\vec{v}$ are the magnitude of the vectors respectively. This dot product will be maximum when $cos\theta$ will be maximum, which occurs when $\theta = 0$ i.e both the vector overlap with each other.\
       Since, $max |\vec{u}|.|\vec{v}|.cos\theta$\
       $= |\vec{u}|.|\vec{v}|.1$\
       $= |\vec{u}|.1$\
       The vector $v$ of unit length to yield max dot product with $u$ is given by 
       $\vec{u}.\vec{v}$ = $= |\vec{u}|$\
       $$\vec{v} = \frac{|\vec{u}|}{\vec{u}}$$
       
2. Outer product
    1. [E] Given two vectors $a = [3, 2, 1]$ and  $b = [-1, 0, 1]$. Calculate the outer product $a^Tb$?\
       [A] Given \
       $$a = [3, 2, 1],    b = [-1, 0, 1]$$\
       Their outer product is given as $a \times b$ which is also known as the 'cross product' given by $a^Tb$

```math
a^Tb = \begin{bmatrix}3\\2\\1\end{bmatrix} \times \begin{bmatrix}-1&0&1\end{bmatrix} = \begin{bmatrix}-3&0&3\\-2&0&2\\-1&0&1\end{bmatrix}
```
<!-- $$A = \left[a \atop a \atop a \atop a\right]$$ -->
a
     2. [M] Give an example of how the outer product can be useful in ML.\
       [A] Following are the use cases where the outer product of the vectors can be useful in ML.
       1. Measure orthogonality of 2 vectors: Two vectors are said to be orthogonal of the angle between them is $90\degree$ and the outer product among them is maximum.
       2. Correlation between 2 vectors
      
          
3. [E] What does it mean for two vectors to be linearly independent?\
   [A] Linearly independent vectors are orthogonal to each other. In such a situation, angel between the vectors will be $90\degree$. Which means their outer product will be zero.\
   $$\vec{u}\times\vec{v} = |\vec{u}|.|\vec{v}|.sin\theta = 0$$

   
5. [M] Given two sets of vectors $A = {a_1, a_2, a_3, ..., a_n}$ and $B = {b_1, b_2, b_3, ... , b_m}$. How do you check that they share the same basis?\
   [A] In order to check whether two sets of vectors share the same basis, we need to find out the number of independent vectors in their respective vector spaces and compare if they are equal. The set of independent vectors is given by the rank of their augmented matrices.

   $A = {a_1, a_2, a_3, ..., a_n}$ can be written as 
   
```math
A = \begin{bmatrix}a_1\\a_2\\a_3\\a_n\end{bmatrix}
```

   Rank can be found by iteratively performing linear transformation among the rows till we achieve non zero rows those will be the linearly independent rows of the matrix and will be equal to the rank of this matrix.
   
7. [M] Given $n$ vectors, each of $d$ dimensions. What is the dimension of their span?\
   [A] Dimension of the span of $n$ vectors is given by the rank of their augmented matrix.

   
9. Norms and metrics
	1. [E] What's a norm? What is $L_0, L_1, L_2, L_{norm}$?\
   [A]
        1. $L_0$ norm: Number of non zero elements in a vector.
        2. $L_1$ norm: Sum of absolute value of vector elements.\
           $|x|_1 = \sum{|x_i|}$
        4. $L_2$ norm: Length of vector in Euclidean space.\
           $||x||_2 = (\sum{x_i^2})^{\frac{1}{2}}$
        5. $L_{\infty}$ norm: Maximum absolute value of vector elemets.\
           $||x||_{\infty} = max{|x_i|}$
	1. [M] How do norm and metric differ? Given a norm, make a metric. Given a metric, can we make a norm?\
    [A]
    
<!-- Segment break -->

# 5.1.2 Matrices

1. [E] Why do we say that matrices are linear transformations?\
   [A] A linear transformation is a function $g: R^m \rightarrow R^n$ that maps a vector space of dimension $m$ to another vector space of dimension $n$. A linear transformation satisfies following propoerties
   1. Homogenous\
      $g(cx) = cg(x)$
   3. Additive\
      $g(x+y) = g(x) + g(y)$

    When a matrix of dimension $A_{n\times m}$, is multiplied with a vector $v$ of dimension $n$ it results in another vector $u$ of dimension $m$, which inherently a transformation process. 
    $$A_{n\times m}v = u$$
    This transformation via matrices follows the above properties of linearity. Hence matrices are known as linear transformations.
   
2. [E] What’s the inverse of a matrix? Do all matrices have an inverse? Is the inverse of a matrix always unique?\
   [A] Inverse of a matrix $A$ is another matrix $A^{-1}$ such that when multiplied together it yields an identity matrix.
  $$AA^{-1} = I$$
    Identity matrix can be obtained by
   $$A^{-1} = \frac{1}{|A|}Adj(A)$$
   Where $|A|$ is the determinant and $\frac{1}{|A|}Adj(A)$ is the adjoint. For the matrices where the determinant is $0$ the inverse is not defined. Such matrices are also known as *singular matrices*.\
   Inverse of a matrix is always unique.
3. [E] What does the determinant of a matrix represent?\
   [A] The determinant of the matrix is given by
   $$det(A) = |A| = \sum_{i=1}^{n}A_{ij}C_{ij}$$ where $C_{ij}$ is the cofactor matrix. Determinant is only defined for square matrices. It represents the area/volume enclosed between the vectors.
   
5. [E] What happens to the determinant of a matrix if we multiply one of its rows by a scalar $t \times R$?\
   [A] As per the row scaling property, on multiplying one of the rows with a scalar $t$ the resulting determinant will be scaled by $t$. 
   $$det(t\times A) = t\times det(A)$$
   
6. [M] A $4 \times 4$ matrix has four eigenvalues $3, 3, 2, -1$. What can we say about the trace and the determinant of this matrix?\
   [A] Trace of a matrix is given by the sum of it's eigen values.\
   Trace of the matrix : For a diagonal matrix, the sum of all it's diagonal elements.
   $$Tr(A) = \sum_{i=1}^{n}a_{ij}$$
   Eigen values and eigen vectors: For the given relation
   $$AX = \lambda X$$
   $\lambda$ is the eigen value of $A$ and $X$ is eigen vector of A given $X != 0$. The charecteristic equation of $A$ is given by
   $$det(A-\lambda I)=0$$ whose roots give the eigen values of the Matrix $A$ as $\lambda _1$, $\lambda _2$, .. $\lambda _n$
   
8. [M] Given the following matrix:<br>
```math
	\begin{bmatrix}1&4&-2\\-1&3&2\\3&5&-6\end{bmatrix}
```

	Without explicitly using the equation for calculating determinants, what can we say about this matrix’s determinant?
	**Hint**: rely on a property of this matrix to determine its determinant.
   [A] The determinant of this matrix is $0$.
   We can perform some linear transformation to the columns and observe one of the columns can become all $0$s which will make the determinant to be $0$.
```math
   \begin{bmatrix}1&4&-2\\-1&3&2\\3&5&-6\end{bmatrix} = -2\times \begin{bmatrix}1&4&1\\-1&3&-1\\3&5&3\end{bmatrix} = \begin{bmatrix}1&4&0\\-1&3&0\\3&5&0\end{bmatrix}
```
10. [M] What’s the difference between the covariance matrix $A^TA$ and the Gram matrix $AA^T$?

12. Given $A \in R^{n \times m}$ and $b \in R^n$
	1. [M] Find $x$ such that: $Ax = b$.
	1. [E] When does this have a unique solution?
	1. [M] Why is it when A has more columns than rows, $Ax = b$ has multiple solutions?
	1. [M] Given a matrix A with no inverse. How would you solve the equation $Ax = b$? What is the pseudoinverse and how to calculate it?

13. Derivative is the backbone of gradient descent.
	1. [E] What does derivative represent?
	1. [M] What’s the difference between derivative, gradient, and Jacobian?

14. [H] Say we have the weights $w \in R^{d \times m}$ and a mini-batch $x$ of $n$ elements, each element is of the shape $1 \times d$ so that $x \in R^{n \times d}$. We have the output $y = f(x; w) = xw$. What’s the dimension of the Jacobian $\frac{\delta y}{\delta x}$?
15. [H] Given a very large symmetric matrix A that doesn’t fit in memory, say $A \in R^{1M \times 1M}$ and a function $f$ that can quickly compute $f(x) = Ax$ for $x \in R^{1M}$. Find the unit vector $x$ so that $x^TAx$ is minimal.
	
	**Hint**: Can you frame it as an optimization problem and use gradient descent to find an approximate solution?