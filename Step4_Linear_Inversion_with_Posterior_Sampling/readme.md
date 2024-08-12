Step 4 Linear Inversion with Posterior Sampling: 
- A GUI tool to interactively generate the true model
- Bayesian Least-Squares Method (BLSM) with Posterior Sampling based on Cholesky decomposition
- Bayesian Non-negative Least-Squares Method (BNNLSM) with Bounded Metropolis-Hastings Algorithm for Posterior Sampling
- Bayesian Bouded Least-Squares Method (BBLSM) with Bounded HMC Algorithm for Posterior Sampling
- Bayesian Joint Data Inverison 

---

## Further Understanding of [HMC](https://en.wikipedia.org/wiki/Hamiltonian_Monte_Carlo) Algorithm：

Hamiltonian Monte Carlo (HMC) is a Markov Chain Monte Carlo (MCMC) method designed to efficiently sample from complex probability distributions, especially in high-dimensional model spaces. Traditional MCMC methods, such as the Metropolis-Hastings algorithm, often struggle with slow mixing and inefficient exploration in these settings due to the random-walk behavior. HMC overcomes these issues by introducing concepts from Hamiltonian dynamics to guide the sampling process. 

### Mathematical Background of  A Conservative Field

In a **conservative field** (a vector field), the work done in moving an object from one point to another is independent of the path taken between the two points. In other words, the work only depends on the initial and final positions, not on how you get from one to the other. 3 key properties of conservative fields:

- **Path Independence**: In a conservative field, the work done by the field on an objective moving from point A to point B is the same regardless of the path taken between these two points. The work done by the field when moving an object from one point to another is equal to the decrease in potential energy:
```math
W=-\Delta U=U\left(\mathbf{r}_A\right)-U\left(\mathbf{r}_B\right)
```

- **Conservation of Mechanical Energy**: In a conservative field, mechanical energy is conserved. The total mechanical energy of an object, which is the sum of its kinetic energy $K$ and potential energy $U$, remains constant as the object moves:
```math
E_{\text {total }}=K+U=\text { constant }
```

- **Curl-free**: A conservative field is one in which the curl of the field is zero, which means that the force $`F`$ acting on an object is directed **opposite** to the direction in which the potential energy increases most rapidly:
```math
\nabla \times \mathbf{F}=0
```
```math
\mathbf{F}=-\nabla U
```

Two common conservative fields: (1) Gravitational Field;  (2) Electrostatic Field.

### Relationships between Position $\mathbf{m}$,  Velocity $\mathbf{v}$, Momentum $\mathbf{p}$, Potential Energy $U(\mathbf{m})$, and Kinetic Energy $K(\mathbf{p})$

The definition of Kinetic Energy $`K(\mathbf{p})`$ and Momentum $`\mathbf{p}`$ are:
```math
K = \frac{1}{2} mv^2
```
```math
p = mv \quad \Leftrightarrow \quad  v = \frac{p}{m}
```

Note $`m`$ in the above equation represents the **mass** when it is a scalar. So we can obtain the matrix form of the Kinetic Energy $`K(\mathbf{p})`$ with respect to Momentum $`\mathbf{p}`$ and Mass Matrix $`\mathbf{M}`$:
```math
K = \frac{1}{2} m (\frac{p}{m})^2 = \frac{p^2}{2m} \quad \text{Matrix form:}\ K(\mathbf{p}) = \frac{1}{2}\mathbf{p}^T \mathbf{M}^{-1} \mathbf{p}
```

The relationship between Position $`\mathbf{m}`$, Velocity $`\mathbf{v}`$, and Kinetic Energy $`K(\mathbf{p})`$:
```math
\frac{d \mathbf{m}}{dt} = \mathbf{v} = \frac{d K(\mathbf{p})}{d\mathbf{p}} = \frac{d (\frac{\mathbf{p}^2}{2m})}{d\mathbf{p}} = \frac{\mathbf{p}}{m} = \mathbf{v}
```

The relationship between Momentum $\mathbf{p}$ and Potential Energy $`U(\mathbf{m})`$:
```math
\frac{d\mathbf{p}}{dt} = \frac{d(m\mathbf{v})}{dt} = m\frac{d\mathbf{v}}{dt} = m\mathbf{a} = \mathbf{F} = -\nabla U(\mathbf{m})
```

### Understanding of the HMC Algorithm

The HMC algorithm abstracts each possible point in the model space as the position $`\mathbf{m}`$ in Hamiltonian Dynamics. The expression for the potential energy $`U(\mathbf{m})`$ at each point is defined by the **misfit function** corresponding to that point. The exploration process in the model space is analogous to the movement of an object in Hamiltonian Dynamics and can be described by the following two dynamic equations:
```math
\text{Position update:}\quad \frac{d \mathbf{m}}{d t}=\frac{\partial H(\mathbf{m}, \mathbf{p})}{\partial \mathbf{p}}=\frac{\partial K(p)}{\partial \mathbf{p}}=\mathbf{M}^{-1} \mathbf{p}
```

```math
\text{Momentum update:}\quad \frac{d \mathbf{p}}{d t}=-\frac{\partial H(\mathbf{m}, \mathbf{p})}{\partial \mathbf{m}}=-\frac{\partial U(m)}{\partial \mathbf{m}}=-\nabla U(\mathbf{m})
```

The exploration process in HMC at each state $`(\mathbf{m}, \mathbf{p})`$ involves solving the above two differential equations. In practice, this is done using the **leapfrog** numerical integration method as a substitute. One important point to note: Since the HMC algorithm simulates the movement of an object in a conservative field, the total energy $`H(\mathbf{m}, \mathbf{p}) = U(\mathbf{m}) + K(\mathbf{p})`$ should theoretically be equal at any state $`(\mathbf{m}, \mathbf{p})`$. However, because the above differential equations are solved approximately using a numerical iterative method, there is inevitably some numerical error! Therefore, in practice, when a new state $`(\mathbf{m}_{\text{new}}, \mathbf{p}_{\text{new}})`$ is obtained after once leapfrog integration, the energy is not perfectly conserved. In this case, the Metropolis rule is introduced to evaluate whether the energy difference between the two states is too large: if the energy difference is too large, the new state is rejected; if the energy difference is acceptable, the new state is accepted. The energy difference measured by the Metropolis rule in the algorithm is used to calculate the acceptance rate.

Based on the above development, we can understand that the key hyperparameters affecting the acceptance rate of the HMC algorithm are related to Leapfrog integration, specifically:

- `leapfrog_size`: The number of iterations used in each leapfrog integration to approximate the solution of the two differential equations at the current state.
- `step_size`: The step size for each iteration, which can also be regarded as the exploration step.

A good combination of hyperparameters can ensure that the acceptance rate in HMC sampling remains above 30% most of the time. Another important point to note: The purpose of HMC is to collect a large number of samples to help us understand the posterior distribution, not to find the maximum likelihood point of the posterior! **HMC is not solving an optimization problem**! It is simply a more efficient sampling method than the Metropolis-Hastings Algorithm (with the same total number of samples, HMC generally obtains more accepted samples). Essentially, both methods serve the same purpose. Therefore, since HMC is not solving an optimization problem, even with a large number of iterations, the acceptance rate of each sample will not change significantly. The sample size only affects our final understanding of the posterior distribution.

### A specific case: HMC for a Linear problem

For a linear Bayesian inversion with Gaussian prior, the misfit function is as follows:
```math
\chi(\mathbf{m})=\frac{1}{2}\left(\mathbf{m}-\mathbf{m}^{\text {prior }}\right)^T \mathbf{C}_M^{-1}\left(\mathbf{m}-\mathbf{m}^{\text {prior }}\right)+\frac{1}{2}\left(\mathbf{G m}-\mathbf{d}^{\text {obs }}\right)^T \mathbf{C}_D^{-1}\left(\mathbf{G m}-\mathbf{d}^{\text {obs }}\right)
```

The potential energy is defined as $`U(\mathbf{m}) = \chi(\mathbf{m})`$, so its gradient, and even the Hessian matrix, have deterministic expressions:
```math
\nabla U(\mathbf{m})=\mathbf{C}_M^{-1}\left(\mathbf{m}-\mathbf{m}^{\text {prior }}\right)+\mathbf{G}^T \mathbf{C}_D^{-1}\left(\mathbf{G} \mathbf{m}-\mathbf{d}^{\text {obs }}\right)
```

```math
\mathbf{H}_\chi=\mathbf{C}_M^{-1}+\mathbf{G}^T \mathbf{C}_D^{-1} \mathbf{G}
```

The Kinetic Energy $`K(\mathbf{p})=\frac{1}{2} \mathbf{p}^T \mathbf{M}^{-1} \mathbf{p}`$ is always the same. When the gradient of $`U(\mathbf{m})`$ has a deterministic expression, leapfrog integration becomes straightforward. When the Hessian matrix has a deterministic expression, the mass matrix $`\mathbf{M}`$ can be set as the Hessian matrix. 

### Two Methods for Measuring the Distance Between Two Model Samples

- Difference in $`\chi(m)`$ values (Misfit Function Difference)
```math
\Delta \chi=\chi\left(\mathbf{m}_1\right)-\chi\left(\mathbf{m}_2\right)
```
This measure tells you how much the misfit (or the negative log-posterior when the prior information are all Gaussian distribution) changes between the two samples.   This measure is particularly useful when you are interested in comparing how close the two samples are in terms of their fit to the **observed data** and **prior information**. It reflects the "quality" of the fit for each sample, not just the spatial relationship between them.

- Euclidean Distance in Parameter Space (Spatial Distance)
```math
d\left(\mathbf{m}_1, \mathbf{m}_2\right)=\left\|\mathbf{m}_1-\mathbf{m}_2\right\|=\sqrt{\sum_{i=1}^n\left(m_{1 i}-m_{2 i}\right)^2}  
```
This distance measure tells you how far apart the two samples are in the parameter space. It is a purely geometric measure and does not take into account the likelihood or posterior probability directly. This is useful when you are interested in understanding the **diversity** of the samples in terms of the parameter values themselves. It helps you to see how spread out the samples are in the parameter space, which can be important for assessing the **exploration efficiency** of the HMC algorithm.
