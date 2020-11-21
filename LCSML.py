# Implements a simple LCS that learns to behave as the XOR gate by using an environment of multiplexers.

# In this implementation, a classifier will be represented as a list containing four elements:
# 1. The rule in the classifier.
# 2. The number of learning iterations it has been part of [M].
# 3. The number of learning iterations it has been part of [C].
# 4. Its current fitness, which is the division of (3) / (4).

# DISCLAIMER
# =======================
# The software is provided "As is", without warranty of any kind, express or implied, including but not limited to the warranties of merchantability, fitness for a particular purpose and noninfringement. In no event shall the authors or copyright holders be liable for any claim, damages or other liability, whether in an action of contract, tort or otherwise, arising from, out of or in connection with the software or the use or other dealings in the software.

import random
import numpy
import pandas

# === GENETIC OPERATORS ===

# Implements the one point crossover
def combine(parentA, parentB):
  ruleA = parentA[0]
  ruleB = parentB[0]  
  cPoint = numpy.random.randint(1, len(ruleA))
  offspringA = [''.join(ruleA[0:cPoint]) +  ''.join(ruleB[cPoint:]), 0, 0, 0]
  offspringB = [''.join(ruleB[0:cPoint]) +  ''.join(ruleA[cPoint:]), 0, 0, 0]
  return offspringA, offspringB

# Implements mutation
def mutate(classifier, mRate):
  rule = list(classifier[0])
  for i in range(len(rule) - 1):    
    if (random.random() <= mRate):
      if (rule[i] == '0'):
        if (random.random() < 0.5):
          rule[i] = '1'
        else:
          rule[i] = '#'
      elif (rule[i] == '1'):
        if (random.random() < 0.5):
          rule[i] = '0'
        else:
          rule[i] = '#'
      else:
        if (random.random() < 0.5):
          rule[i] = '0'
        else:
          rule[i] = '1'
  if (random.random() <= mRate):
      if (rule[-1] == '0'):
        rule[-1] = '1'
      else:
        rule[-1] = '0'
  return [''.join(rule), 0, 0, 0]

# implements tournament selection
def select(set, tournamentSize):
  winner = numpy.random.randint(0, len(set))
  for i in range(tournamentSize - 1):
    rival = numpy.random.randint(0, len(set))
    if (set[rival][3] > set[winner][3]):
      winner = rival
  return set[winner]

# === CLASSIFIER FUNCTIONS ===

# Verifies if a rule matches an instance
def match(rule, instance):
  for i in range(len(instance) -1):
    if (rule[i] != '#') and (rule[i] != instance[i]):
      return False
  return True

# Implements covering
def covering(instance):
  rule = list(instance)
  for i in range(len(rule)):
    if (random.random() < 0.1):
      rule[i] = '#'
  rule.append(instance[-1])
  return [''.join(rule), 0, 0, 0]

# Splits [M] into correct [C] and incorrect [I] sets
def split(M, instance):
  C = []
  I = []
  for classifier in M:
    rule = classifier[0]    
    if (rule[-1] == instance[-1]):
      C.append(classifier)
    else:
      I.append(classifier)
  return C, I

# Tests a set of classifiers on a given environment
def test(P, E):
  accuracy = 0.0
  for instance in E:
    M = []
    expectedOutput = instance[-1]
    for classifier in P:
      if match(classifier[0], instance):
        M.append(classifier)
    result = "INCORRECT"    
    if len(M) > 0:
      best = M[0]
      for classifier in M:
        if classifier[3] > best[3]:
          best = classifier      
      rule = best[0]      
      if expectedOutput == rule[-1]:
        result = "CORRECT"
        accuracy = accuracy + 1
      print(str(instance) + " -> " + expectedOutput + " / " + str(rule) + " -> " + str(rule[-1]) + ", " + str(result))
    else:
      print(str(instance) + " -> " + expectedOutput + " / ? -> ? , " + str(result))
  print("==================================")
  print("Accuracy: " + str(accuracy / len(E)))

# === THE LCS ALGORITHM ===

# Implements a simple LCS to solve the problem
def lcs(n, cycles, E, mRate):
  random.shuffle(E)  
  k = 0
  P = []
  for i in range(cycles):
    instance = E[k]
    k = k + 1
    if (k >= len(E)):
      k = 0
    M = []
    for classifier in P:
      rule = classifier[0]
      if match(rule, instance):        
        M.append(classifier)      
    if len(M) == 0:
      classifier = covering(instance)
      M.append(classifier)
      P.append(classifier)
    C, I = split(M, instance)
    for classifier in C:
      classifier[1] = classifier[1] + 1
      classifier[2] = classifier[2] + 1
      classifier[3] = classifier[2] / classifier[1]
    for classifier in I:
      classifier[1] = classifier[1] + 1    
      classifier[3] = classifier[2] / classifier[1]
    if len(C) == 0:
      C = P    
    parentA = select(C, 2)
    parentB = select(C, 2)
    offspringA, offspringB = combine(parentA, parentB)
    mutate(parentA, mRate)
    mutate(parentB, mRate)
    P.append(parentA)
    P.append(parentB)
    random.shuffle(P)
    while len(P) > n:
      worst = P[0]
      for classifier in P:
        if classifier[3] < worst[3]:
          worst = classifier
      P.remove(worst)
  return P

# Defines the environment
#dataset = "bWeather.csv"
dataset = "bTitanic.csv"
data = pandas.read_csv(dataset)
data.to_string()
listofstr = data.astype(str)
E = list(map(lambda x: ''.join(x), listofstr.values))

# Only split when dealing with dataset bTitanic
random.shuffle(E)
ETrain = E[0:1320]
ETest = E[1320:]

# Runs the LCS
classifiers = lcs(500, 2500, ETrain, 0.1)

# Tests the performance of the classifiers
test(classifiers, ETest)
