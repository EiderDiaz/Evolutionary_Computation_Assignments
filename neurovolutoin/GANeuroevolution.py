# Solves a collection of deceptive functions by using a simple genetic algorithm with binary representation. 

# DISCLAIMER
# =======================
# The software is provided "As is", without warranty of any kind, express or implied, including but not limited to the warranties of merchantability, fitness for a particular purpose and noninfringement. In no event shall the authors or copyright holders be liable for any claim, damages or other liability, whether in an action of contract, tort or otherwise, arising from, out of or in connection with the software or the use or other dealings in the software.

import random
import numpy
import math
import matplotlib.pyplot

def logsig (x):
  a = numpy.zeros((len(x), 1))
  for i in range(len(x)):     
    a[i] = [1 / (1 + math.exp(-x[i]))]
  return a

logsig([-10, 0, 10])

# Implements the random initialization of individuals using the real-valued representation.
def createIndividual(n):
  return numpy.random.random([3 * n + 1])

# Implements the one point crossover on individuals using the real-valued representation.
def combine(parentA, parentB, cRate):
  if (random.random() <= cRate):
    cPoint = numpy.random.randint(1, len(parentA))   
    offspringA = numpy.append(parentA[0:cPoint], parentB[cPoint:])
    offspringB = numpy.append(parentB[0:cPoint], parentA[cPoint:])
  else:
    offspringA = numpy.copy(parentA)
    offspringB = numpy.copy(parentB)
  return offspringA, offspringB

# Implements mutation using the real-valued representation.
def mutate(individual, mRate):
  for i in range(len(individual)):
    if (random.random() <= mRate):
      if (numpy.random.random() < 0.5):
        individual[i] += numpy.random.random()
      else:
        individual[i] -= numpy.random.random()
  return individual

# Implements the fitness function
def evaluate(individual, n):
  evaluation = 0.0  
  w1 = individual[0:n]
  w1 = w1.reshape(n, 1)
  b1 = individual[n:2 * n]
  b1 = b1.reshape(n, 1)
  w2 = individual[2* n:3*n]
  b2 = individual[3 * n]
  p = numpy.arange(-3, 3.01, 0.1)
  t = p ** 3
  for i in range(len(t)):
    a1 = logsig(numpy.add(numpy.matmul(w1, p[i].reshape(1, 1)), b1))  
    a2 = numpy.add(numpy.matmul(w2, a1), b2) * 10
    evaluation += (math.pow(a2[0] - t[i], 2))
  return evaluation / len(t);

# Implements the tournament selection.
def select(population, evaluation, tournamentSize):
  winner = numpy.random.randint(0, len(population))
  for i in range(tournamentSize - 1):
    rival = numpy.random.randint(0, len(population))
    if (evaluation[rival] < evaluation[winner]):
      winner = rival
  return population[winner]

# Plots a solution.
def plot(individual, n):  
  w1 = individual[0:n]
  w1 = w1.reshape(n, 1)
  b1 = individual[n:2 * n]
  b1 = b1.reshape(n, 1)
  w2 = individual[2* n:3*n]
  b2 = individual[3 * n]
  p = numpy.arange(-3, 3.01, 0.1)
  y = numpy.zeros(len(p))
  for i in range(len(y)):
    a1 = logsig(numpy.add(numpy.matmul(w1, p[i].reshape(1, 1)), b1))  
    a2 = numpy.add(numpy.matmul(w2, a1), b2) * 10   
    y[i] = a2
  matplotlib.pyplot.plot(p, y, label = "GA Approximation")
  matplotlib.pyplot.plot(p, p**3, label = "Objective")  
  matplotlib.pyplot.legend()
  matplotlib.pyplot.title("GA Approximation vs. Objective")
  matplotlib.pyplot.show()

# Implements a genetic algorithm.
def geneticAlgorithm(n, populationSize, generations, cRate, mRate):
  # Creates the initial population (it also evaluates it)
  population = [None] * populationSize
  evaluation = [None] * populationSize  
  for i in range(populationSize):
    individual = createIndividual(n)
    population[i] = individual
    evaluation[i] = evaluate(individual, n)
  # Keeps a record of the best individual found so far
  index = 0;
  for i in range(1, populationSize):
    if (evaluation[i] < evaluation[index]):
      index = i;
  bestIndividual = population[index]
  bestEvaluation = evaluation[index]
  best = [0] * generations
  avg = [0] * generations
  # Runs the evolutionary process    
  for i in range(generations):
    k = 0
    newPopulation = [None] * populationSize    
    for j in range(populationSize // 2):
      parentA = select(population, evaluation, 3)
      parentB = select(population, evaluation, 3)
      newPopulation[k], newPopulation[k + 1] = combine(parentA, parentB, cRate)       
      k = k + 2    
    population = newPopulation
    for j in range(populationSize):
      population[j] = mutate(population[j], mRate)
      evaluation[j] = evaluate(population[j], n)
      # Keeps a record of the best individual found so far      
      if (evaluation[j] < bestEvaluation):        
        bestEvaluation = evaluation[j]
        bestIndividual = population[j]
    best[i] = bestEvaluation
    avg[i] = numpy.average(evaluation)  
  matplotlib.pyplot.plot(range(generations), best, label = "Best")
  matplotlib.pyplot.plot(range(generations), avg, label = "Average")
  matplotlib.pyplot.legend()
  matplotlib.pyplot.title("GA Run")
  matplotlib.pyplot.show()
  return bestIndividual, bestEvaluation

# Runs the genetic algorithm
n = 5
#solution, evaluation = geneticAlgorithm(n, 100, 200, 0.9, 0.05)
solution, evaluation = geneticAlgorithm(n, 100, 100, 0.9, 0.10)
print(solution)
print(evaluation)
plot(solution, n)
