# Replicating Optimization Results in Practice

Numerical optimization problems are found at the heart of most (if not all) of the statistical and machine learning problems that we encounter. In general, we don't need to think about it because we don't need to look too closely at the numerical properties of the problem being solved. However, I've encountered several cases in school and work where it was necessary where I dig deeper than simply solving an optimization problem by attempting to verify the results of one problem against another.

It's difficult because the correct implementation requires that we ensure that the problem we are solving is the same one (IE, no mistakes were made), that the starting points are the same, and that we arrive at a similar solution. In some cases, we may also need to make sure that the solver being used has the same or similar numerical properties.

There are some key points which I've found to greatly shorten the total amount of work necessary to perform the task.

## Define what the goal of the replication is.

Scoping the problem can lead to time savings just because it can let you know what steps to avoid. For instance, if you are just interested in vague correctness, you can hit the problem with a sub-optimal but easy to implement solver (IE stochastic gradient descent). Are the results being used to replicate some other part of a larger framework? In that case, solving the problem may not suffice - we may have to get exactly what the original results were.

## Simplify the problem.

This suggestion is added with an eye towards constrained optimization. In grad school, I often heard the advice that the first step to working on a constrained problem is to transform it into an unconstrained dual problem. At that point, we can apply a vast array of solvers to get our answer.

## Drawing from constrained optimization: Using barrier functions for pinpoint replication.

I haven't implemented this strategy but it occurred to me during a recent project. Above, I noted we may need to replicate a larger framework where the optimization results are just a component. In this case, *it may be more important to get the same function value*. If information about the numerical method and tolerances is missing, this can be trying.

**To avoid this, we can make function evaluation a part of the optimization problem**. We can take one tool from constrained optimization called a [barrier function](https://en.wikipedia.org/wiki/Barrier_function). This will introduce a term to cause the size of the function to explode when approaching the boundary. We can apply boundary functions to restrict the domain of reasonable solution to a small area around the original results to force the solver to come to the same value. From there, whatever output from the solver that is used elsewhere can be evaluated against the original results.
