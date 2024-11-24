# Brainstorming - Empirical Example II
## Starting Point
+ The vast majority of Bayesian optimization case studies employs GPs :point_right: it is critical to feature them extensively, outside of the domain of biotech
+ as general considerations are explained, the characteristic requirements of biotech should be featured
+ all these specific choices should find place in our 2nd case study



## Goal Posts for the Empirical Example
+ __two continous design parameters__, __pH__ should be included as the __region of optimality is intuitive__ which aides illustrating the __need for Bayesian stats modeling__

+ TEMPERATURE IS A GREAT SECOND PARAMETER


+ initial experimental designs with __space filling regime__ plus __log-transformation__ (2nd design parameter could be concentration)
+ KPI should be relatable / straight forward 
+ __experiments conducted with MTP__, highlighting effects / key points such as
    + batch effect
    + column-wise time delay during pipetting
    + single row with systematic offset, pipette needle is off
    + randomizing wells to reduce influence of systematic effects, however, not affordable plus might not be feasible due to experimental constraints
+ possibility to compare with traditional surrogate modeling struggling to capture the trend

## Story Points
+ goal: maximize rate constant
+ available data: time series :point_right: product concentration (heteroskedatic noise), 
+ Naive approach following best practices in empirical example I
    + linearize timeseries :point_right: rate constants
    + appropriate choices for GP initialization (kernel selection, ls priors, etc.)
    + :point_right: GP fits great, histogram with TS proposals is counterintuitive
+ Surrogate model only as good as we allow it to be
+ discuss 

### Key Figures
+ comparison of naive GP w/ TS proposals and Bayesian process model powered GP
+ visualization of model (more accessible, simplified computation graph)
+ optional:
    + residual heatmap highlighting time delays and pipetting row effect