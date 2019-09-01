# Optimization algorithms

### Travelling Salesman Problem + Function Approximation

**ES** ( evolution strategies )
  - randn ( normal dist ) in techniques (crossover + mutation + ..) cause evolution tree quite diverge from fittest group
  - in TSP is one of the key to swap(src, dst, i) only one city instead of whole sequence of cities ( gene )
  - tournament for ES strategies is important priority randomization for choosing what to evolve from population

pros | cons
--- | ---
usually it works | usually it works
should be number 1 to try | not always optimal
easy to implement and define operations | slow in big domains

**ACO** ( ant colony ) 
  - for TSP need to redefine distance matrix, as "mu +- sigma" should represent move in closest cities ( in contrast to cities ids )
  - important keeping archive with top trails, and then sort others but in a evaporation manner, aka if some trail scent becomes weak need to evaporate by FIFO from point it is too week, not from point where it was added
  - seems in addition to pheromones need to use heuristic at least in discrete domains (TSP : dist to cities from current one ), hard to say if this is feature ~ but likely it is therefore seems more benefits in discrete problems
  - while idea is quite descriptive in discrete domain, it is very different from continous spaces implementation and quite counter intuitive :
    - [discrete implementation](https://github.com/rezer0dai/optim/blob/master/tsp/aco.py)  for TSP, [continous implementation](https://github.com/rezer0dai/optim/blob/master/tsp/acob.py)  for TSP, [continous implementation](https://github.com/rezer0dai/optim/blob/master/fn/acoc.py)  for function approximation

pros | cons
--- | ---
appears to works quite nicely | performance overhead to looking for sigma + draw from distribution
tabular cases (TSP) quite efficient | convergence is based on keeping well balance archive, but usually requires add heuristic as part of pheromones scent
in continuous cases highly depends on archiving technique which can be tuned as well | probabilistic model, not always what you looking for


**PSO** ( particle swarm ) 
  - cognitive vs social are randomized but scaled by degree of importanance ( [rand > cog_importance] * error or w - cog_importance * rand * error  )
  - easy to find local minima and never goes out, thats why some sort of mutation should be used
  - mutation i use is simulated annealing, though only in TSP discrete domain, as in continous w - cog_importance * error * rand will give use mutation around w by default

pros | cons
--- | ---
potent methodology, reassemble backprop  | lots of hyperparametrs to tune
fast and systematic convergence | easy to converge to local minima
intuitive idea in continuous problems | harder to comprehend for discrete ones
. |  lack of systematic mutation
  

**NES** ( neural evolution strategies )
  - sigma need to be multiplied and divided separately due to matrix operations
  - nice connection / interpolation between : ANN - NES - BNN - NoisyNetworks

pros | cons
--- | ---
fast | sample not so eficient
approximating gradient wrt noise not input | multilayered seems harder to converge
systematic updates in parameters space | not best for optimization in any domain, but should be quite efficient in RL settings ( policy optimizaiton )


**SA** ( simulated annealing )
  - division by temperature is inside np.exp , easy to omit that :) !! np.exp((c(old)-c(new))/temp)

pros | cons
--- | ---
guaranted to converge given enough samples | sample uneficient
systematic approach to jump out from local minima | only mutation technique used
easy to implement and combine with other approaches | 

### random notes 
    - pay attention : uniform vs normal :: scale ( 0..100% ) vs oscilate ( -, + + scale)
    - sigma is abs distance, noise can oscilate and scale sigma

## references

TPO : 
- https://pdfs.semanticscholar.org/d4d3/fa4eadc8b3c4b5989960688a3f2833c984cf.pdf
- https://github.com/grosa1/pso_travelling_salesman/blob/master/main.py
- https://github.com/marcoscastro/tsp_pso/blob/master/tsp_pso.py

ACO:
- https://github.com/rhgrant10/acopy/blob/master/acopy/solvers.py

ES :
- https://towardsdatascience.com/evolution-of-a-salesman-a-complete-genetic-algorithm-tutorial-for-python-6fe5d2b3ca35
- https://github.com/ezstoltz/genetic-algorithm/blob/master/genetic_algorithm_TSP.ipynb
- https://courses2.cit.cornell.edu/cs5724/schedule.htm
- https://towardsdatascience.com/gradient-descent-vs-neuroevolution-f907dace010f
- https://eng.uber.com/deep-neuroevolution/

NES : 
- https://openai.com/blog/evolution-strategies/
- https://gist.github.com/karpathy/77fbb6a8dac5395f1b73e7a89300318d

SA :
- https://ericphanson.com/blog/2016/the-traveling-salesman-and-10-lines-of-python/

mix : 
- https://github.com/guofei9987/scikit-opt


