---
title: "A Quick Look at APC"
author: "Peter Caya"
date: "7/6/2019"
output: 
  html_document:
    keep_md: yes
---
  
 Age-period-cohort analysis is a time series methodology used to describe the relationship between some phenomena (like cancer rates, or defaults on loans) by using three variables:

* Age - This could be the age of a cancer patient or how many months a loan has existed.
* Cohort - Unique qualities shared by specific groups as they move through time. Generally this is a birth cohort or in the case of  initiation cohort. In our cancer example, we may compare baby boomers to Gen Xers.
* Period - External factors affecting all age groups. 

Reducing the model to these three variables provides information about the independent affect each has on the dependent variable. A graphical example of the intuition is available in figures 1 and 2 of [Smith and Wakefield's paper](http://faculty.washington.edu/jonno/papers/smith-wakefield-16.pdf). The charts give the age, cancer type, birth cohort, and period from the Danish cancer study. It shows what we would expect: A higher age generally seems to lead to a higher incidence of cancer. Older cohorts also seem to be more exposed to cancer risk.  

The example below uses data from the [apc package](https://cran.r-project.org/web/packages/apc/index.html) in R. The actual data originates from [Clayton and Schifflers' paper](https://www.ncbi.nlm.nih.gov/pubmed/3629047) and *Cancer Incidence in Five Continents*. It displays the  frequency for rates of lung cancer in Belgium: 

![](/img/hello_world.jpeg)<!-- -->

We can see what we might expect: Someone who is younger, all things held equal, has a lower chance of lung cancer. Interestingly, there was an uptick in cancer occurrences in the 1965-1969 for the 25-29 age group.

A simple example of an APC model is:
$$ln(y_{ijk}) = \sigma+\alpha_i+\beta_j+\gamma_k $$

Where:

* $y_{ijk}$ is the number of instances for age $i$ and time period $j$.
* $\sigma$ is a constant
* $\alpha$ is the age effect.
* $\beta_j$ is the effect for the period.
* $\gamma_k$ is the effect of the cohort.

# The Identification Problem
 
This is appealing from an explanatory perspectives, but introduces a big problem: The age and period in the analysis will be colinear:

$$ Cohort = Period - Age$$

For example, if one knows the age group an observation occurs from, and the period that the data is from, then the birth cohort for the observation is implied.

As it turns out, there is a fairly deep subfield in determining what the appropriate way for compensating this multicollinearity is. [Columbia University's tutorial on the topic](https://www.mailman.columbia.edu/research/population-health-methods/age-period-cohort-analysis) has a good summary of the methods for coping with the collinearity which I've quickly described below:

* The most simple method is to explore whether one of the variables can be feasibly omitted.
* Introducing a proxy variable for one of the covariates.
* Apply nonparametric functions the factors.
* Apply constraints to the regression analysis.

I'll investigate these methods in a future post. I've noticed from the papers that I've read that a significant portion of the analysis goes into identifying the proper technique from the list above, customizing it for the particular situation, and *then* fitting the model. A modeler may also include effects aside from those described above. In the case of estimating loan default probability, a modeler might include the current FICO score of the borrower and the unemployment rate.

# Overview

APC is a very intuitive modeling strategy but from the reading I've been doing about it, it seems prone to failing unless a large amount of work is applied to ensure that the problem is well-specified (IE - that it avoids the identification problem).


