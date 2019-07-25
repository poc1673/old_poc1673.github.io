---
title: "Automatic R Markdown"
author: "Peter Caya"
date: "July 21, 2019"
output:
  html_document:
    keep_md: yes
---


R Markdown allows one to quickly prototype and automate analysis and then embed it in a document. Recently, I ran into a problem: If I repeat a set of summaries many different data-sets and want to summarize the results, how can I iterate through them without a large amount of copying and pasting?

I was surprised to find relatively few guides on this topic on Stack Overflow or Reddit. The best example I could find was located [here](https://stackoverflow.com/questions/21729415/generate-dynamic-r-markdown-blocks).

I've decided to take the post I linked above and provide a more involved example based on my small project from work.

For this example I'll perform some summary tests and plots based on the mtcars data-set:

1. Generate regression models for mpg.
2. Create a function to summarize the results (a summary table and a residual plot in this case).
3. Generate a list models and output them automatically (no copying or pasting).

Let's begin!

Below, I've written functions to generate the formulas, plots, and regression summaries:

```r
library(pacman)
p_load(knitr,rmarkdown,broom,ggplot2)

gen_formulas <- function(dep,indep){as.formula(paste(dep,paste(indep,collapse = "+"),sep = "~"))}

reg_plot <- function(dep,indep,dat){  return(ggplot(dat,aes_string(x = dep,y = indep))+geom_point()+stat_smooth(method = "lm",col = "blue")+theme_bw())}

gen_reg_summaries <- function(dep,indep,dat){
  for_reg <- gen_formulas(dep = dep,indep = indep)
  temp_reg  <- lm(data = dat,formula = for_reg)
  reg_summary <- tidy(temp_reg)
  ret_plot <- reg_plot(dat = dat,dep = dep,indep = indep)
  return(list("Reg" = temp_reg,
              "Summary" = reg_summary,
              "Plot" = ret_plot))}
```
The regression results are then created as a list for each model. These are the values we'll be returning for our report:

```r
formula_info <- c("disp","drat","wt","qsec")

regression_results <- lapply(formula_info, function(x){gen_reg_summaries(dep = "hp",indep = x,dat = mtcars) })
names(regression_results) <- formula_info
```
To avoid the pain of repeatedly copying the results that we will write functions to generate them dynamically.

To do this, we apply the knight_expand function from the knitr package. It's a sort of "meta markdown" function which will allow us to generate the output we seek repeatedly in a loop. In this case, we want to go through each of the results we have so far, and display the table and the plot.

# Simple Example: A single a summary of the regression using HPI:

The code below is fairly simple: We use the knit_expand function. The argument is a text string consisting of a header and a reference to the summary table.  We mark any pieces of code with two curly-braces:

```r
gen_knit_text <- function(num){
  header <- paste('## Regression: {{names(regression_results)[',num,']}}',sep = "")
  smry <- paste('{{regression_results[[',num,']]$Summary}}',sep = "")
  plot <- paste(' ```{r, fig.width = 10 }\n{{regression_results[[',num,']]$Plot}}\n```',sep = "")
  res <- paste(paste(header,smry,plot,sep = "\n"),"\n\n" ,sep = "")
  return(res)
}
```

Now, all we need to do to generate the output is a single call to sapply which will create all of the analysis we were looking for:

```{r, message=FALSE,message=FALSE,warning=FALSE}
text_to_knit <- sapply(X = 1:length(regression_results),FUN = function(x){ knit(text = gen_knit_text(x)) })
```
`r paste(knit(text = knit_expand(text=text_to_knit)),collapse = '\n') `

If you want to see the finished output of this  post in a notebook format, you can see the notebook here. It includes all of the plots and analysis covered in this blog post.
