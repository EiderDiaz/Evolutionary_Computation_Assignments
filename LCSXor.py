# Implements a simple LCS that learns to behave as the XOR gate by using an environment of multiplexers.

# DISCLAIMER
# =======================
# The software is provided "As is", without warranty of any kind, express or implied, including but not limited to the warranties of merchantability, fitness for a particular purpose and noninfringement. In no event shall the authors or copyright holders be liable for any claim, damages or other liability, whether in an action of contract, tort or otherwise, arising from, out of or in connection with the software or the use or other dealings in the software.

import random
import numpy

# Implements the random initialization of a classifier. Here, a classifier will be represented as a list containing four elements:
# 1. The rule in the classifier.
# 2. The number of learning iterations it has been part of [M].
# 3. The number of learning iterations it has been part of [C].
# 4. Its current fitness, which is the division of (3) / (4).
def createClassifier(m):
  rule = [''] * (2 ** m + m + 1)
  for i in range(len(rule) - 1):
    rnd = random.random()
    if (rnd < 0.10):
      rule[i] = '#'
    else:
      if (rnd < 0.55):
        rule[i] = '0'
      else:
        rule[i] = '1'
  if (random.random() < 0.5):
    rule[-1] = '0'
  else:
    rule[-1] = '1'
  return [''.join(rule), 0, 0, 0]

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

# Implements the fitness function
def evaluate(classifier):
  if (classifier[1] > 0):
    classifier[3] = classifier[2] / classifier[1]
  return classifier[3]

# implements tournament selection
def select(set, tournamentSize):
  winner = numpy.random.randint(0, len(set))
  for i in range(tournamentSize - 1):
    rival = numpy.random.randint(0, len(set))
    if (set[rival][3] > set[winner][3]):
      winner = rival
  return set[winner]

# Calculates the output of a multiplexer
def getOutput(instance, m):  
  index = int(instance[0:2], m) + m
  return instance[index]

# Verifies if a rule matches an instance
def match(rule, instance):
  for i in range(len(instance)):
    if (rule[i] != '#') and (rule[i] != instance[i]):
      return False
  return True

# Implements covering
def covering(instance, m):
  rule = list(instance)
  for i in range(len(rule)):
    if (random.random() < 0.1):
      rule[i] = '#'
  rule.append(getOutput(instance, m))
  return [''.join(rule), 0, 0, 0]

# Splits [M] into correct [C] and incorrect [I] sets
def split(M, instance, m):
  C = []
  I = []
  output = getOutput(instance, m)  
  for classifier in M:
    rule = classifier[0]    
    if (rule[-1] == output):
      C.append(classifier)
    else:
      I.append(classifier)
  return C, I

# Tests a set of classifiers on a given environment of multiplexers
def test(P, E, m):
  accuracy = 0
  for instance in E:
    M = []
    expectedOutput = getOutput(instance, m)
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

# Implements a simple LCS to solve the problem
def lcs(n, m, cycles, E, mRate):
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
      classifier = covering(instance, m)
      M.append(classifier)
      P.append(classifier)
    C, I = split(M, instance, m)
    for classifier in C:
      classifier[2] = classifier[2] + 1
      classifier[3] = classifier[3] + 1
      evaluate(classifier)
    for classifier in I:
      classifier[2] = classifier[2] + 1      
      evaluate(classifier)
    if len(C) == 0:
      C = M    
    parentA = select(C, 2)
    parentB = select(C, 2)
    mutate(parentA, mRate)
    mutate(parentB, mRate)
    P.append(parentA)
    P.append(parentB)
    random.shuffle(P)
    while len(P) > n:
      worst = P[0]
      for classifier in P:
        if worst[3] < classifier[3]:
          worst = classifier
      P.remove(worst)
  return P

# Defines the environment
E = [None] * 6
E[0] = "100110"
E[1] = "111010"
E[2] = "110010"
E[3] = "011110"
E[4] = "000011"
E[5] = "101011"

# Runs the LCS
classifiers = lcs(50, 2, 200, E, 0.1)

# Tests the performance of the classifiers
test(classifiers, E, 2)
