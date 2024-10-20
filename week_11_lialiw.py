# -*- coding: utf-8 -*-
"""
Created on Fri Jan 14 16:45:42 2022

@author: Fik
"""

"""
    Sources:
        https://towardsdatascience.com/evolution-of-a-salesman-a-complete-genetic-algorithm-tutorial-for-python-6fe5d2b3ca35
        https://stackoverflow.com/questions/32997395/iterate-through-a-list-given-a-starting-point
    
    Theoretical:
        https://towardsdatascience.com/introduction-to-genetic-algorithms-including-example-code-e396e98d8bf3
        https://towardsdatascience.com/introduction-to-optimization-with-genetic-algorithm-2f5001d9964b
"""

import re, random, operator, math
import pandas as pd
import numpy as np


def read_tsp_data(tsp_name):
    """
    Open the TSP file and put each line cleaned of spaces and newline characters in a list
    Returns a list like:
        ['NAME: ulysses16.tsp', 'TYPE: TSP', 'COMMENT: Odyssey of Ulysses (Groetschel/Padberg)', 'DIMENSION: 16', 'EDGE_WEIGHT_TYPE: GEO', 
         'DISPLAY_DATA_TYPE: COORD_DISPLAY','NODE_COORD_SECTION', '1 38.24 20.42', '2 39.57 26.15', '3 40.56 25.32', ................ 'EOF']
    """
    cleaned = []
    with open(tsp_name) as f:
        content = f.read().splitlines()
        #cleaned = [x.lstrip() for x in content if x != ""]
        cleaned = [x.lstrip() for x in content] #; print(type(cleaned)); print(cleaned)
        pop_el=''
        while pop_el!='EOF':
            pop_el = cleaned.pop()
        #print(type(cleaned)); print(cleaned)
        return cleaned
    
    
def detect_dimension(in_list):
    #Use a regex here to clean characters and keep only numerics
    #Check for the line DIMENSION in the file and keeps the numeric value
    non_numeric = re.compile(r'[^\d]+')#; print(non_numeric)
    for element in in_list:
        if element.startswith("DIMENSION"):
            dimension = non_numeric.sub("",element) #; print(type(dimension))
            return int(dimension)


def detect_start_from(in_list):
    #Use a regex here to clean characters and keep only numerics
    #Check for the line DIMENSION in the file and keeps the numeric value
    start_from=1
    for element in in_list:
        if element.startswith("NODE_COORD_SECTION"):
            break
        start_from+=1
    return start_from


def get_nodes_cords(data, start_from, intFlag, floatFlag):
    """
    Iterate through the list of line from the file if the line starts with a numeric value within the range of 
    the dimension, we keep the rest which are the coordinates of each city 1 33.00 44.00 results to 33.00 44.00
    """
    #print(data)
    aDict={}
    for i in range(start_from, len(data)):
        index, space, rest = data[i].partition(' ')
        #print(rest)
        x, space, y = rest.partition(' ') #; print(i-start_from+1, "x: ", x, " space: ", space, " y: ", y)
        while (x==''):
            x,space,y=y.partition(' ')
    
        #print(i-start_from+1, "x: ", x, " space: ", space, " y: ", y)
        #'''
        if intFlag:
            aDict[i-start_from+1] = [int(x),int(y)]
        if floatFlag:
            aDict[i-start_from+1] = [float(x),float(y)]
        #'''
    #print(aDict)        
    return aDict 


def euclidean(a, b):
    x, y = 0, 1
    return (((a[x]-b[x])**2)+((a[y]-b[y])**2))**0.5


def route_dist(route,coordinates): #starts from 0 to also count distance from last node (-1) to the first node (0)
    dist=0
    for i in range(len(route)):
        dist+=euclidean(coordinates[route[i]],coordinates[route[i-1]])
    return dist
  

def createPopulation(popSize, nodes_list):
    '''
    Parameters
    ----------
    popSize : integer
        number of individuals (TSP solutions) to be produced to create an initial ppopulation
    nodes_list : list
        list of tuples
        1st element of tuple, an integer, represents a node's (city's) name i.e. its id increased by 1
        2nd element of tuple, a list, represents a node's (city's) coordinates
    Returns
    -------
    population_list : list
        A list of dictionaries, each dictionaty represents a TSP solution. 
        Each key of a dictionaty is  a (city) node and its value is the (city) coordinates
        The position (index) in which a TSP solution is found represents its id

    '''
    population_list = []
    for i in range(popSize):
        while(True):
            random_route = random.sample(nodes_list, len(nodes_list))
            if random_route not in population_list:
                population_list.append(dict(random_route))
                break

    return population_list
               

def rank_populationElements(population_list):
    '''
    Parameters
    ----------
    population_list : list
        A list of dictionaries, each dictionaty represents a TSP solution. 
        Each key of a dictionaty is  a (city) node and its value is the (city) coordinates
        The position (index) in which a TSP solution is found represents its id

    Returns
    -------
    TYPE list
        a list of tuples, one for each element, of population_list 
        1st element of tuple represents TSP solution's id
        2nd element of tuple represents TSP solution's score
        
        it is sorted in descending TSP solution's score
    '''
    fitness_dict = {}
    for i in range(len(population_list)):
        route =  [node-1 for node in list(population_list[i].keys())]
        dist = route_dist(route, list(population_list[i].values()))
        fitness_dict[i]=1/float(dist)
    
    return sorted(fitness_dict.items(), key = operator.itemgetter(1), reverse = True)


def selection(popRanked, eliteSize):
    '''
    Parameters
    ----------
    popRanked : list
        list of tuples. 
        1st element of tuple represents TSP solution's id
        2nd element of tuple represents TSP solution's score
    eliteSize : integer
        number of (best) elements, population's individuals, to be selected for sure.

    Returns
    -------
    selectionResults : list
        list of integers, each int represents the id of a TSP solution

    '''
    selectionResults = []
    
    df = pd.DataFrame(np.array(popRanked), columns=["Index","Fitness"])
    #print(df)
    
    df['cum_sum'] = df.Fitness.cumsum()
    df['cum_perc'] = 100*df.cum_sum/df.Fitness.sum()
    #print(df)
    
    #print(type(popRanked)); print(popRanked)
    
    for i in range(0, eliteSize):
        selectionResults.append(popRanked[i][0])
        
    for j in range(0, len(popRanked) - eliteSize):
        pick = 100*random.random()
        for i in range(0, len(popRanked)):
            if pick <= df.iat[i,3]:
            #if pick <= df.iat[i,3] and popRanked[i][0] not in selectionResults:
                selectionResults.append(popRanked[i][0])
                break
    #print(selectionResults)       
    return selectionResults


def create_matingPool(population, selectionResults):
    '''
    Parameters
    ----------
    population : list
        A list of dictionaries, each dictionaty represents a TSP solution. 
        Each key of a dictionaty is  a (city) node and its value is the (city) coordinates
        The position (index) in which a TSP solution is found represents its id
    selectionResults : list
        list containing an id of each solution (population individual) to become a parent
    Returns
    -------
    matingpool : list
        A list of dictionaries, each dictionaty represents a TSP solution and its id is found in selectionResults. 
        Each key of a dictionaty is  a (city) node and its value is the (city) coordinates
    '''
    
    matingpool = []
    #print(len(selectionResults))
    for i in range(0, len(selectionResults)):
        index = selectionResults[i]
        #print(index, population[index])
        matingpool.append(population[index])
    #print(len(matingpool))
    return matingpool


def breed(parent1, parent2):
    '''
    Parameters
    ----------
    parent1 : dictionary
        each one of its keys represents a node (a city) of the TSP problem 
        the value of each key represents the node's coordinates
    parent2 : dictionary
        each one of its keys represents a node (a city) of the TSP problem 
        the value of each key represents the node's coordinates
    Returns
    -------
    offspring1 : dictionary
        1st child of parent1 and parent2 via Order (OX) crossover
        same structure as parents
    offspring2 : dictionary
        2nd child of parent1 and parent2 via Order (OX) crossover
        same structure as parents
    '''

    len_p = len(parent2) # len(parent2) == len(parent1)
    
    offspring1 = list(parent1.items())
    offspring2 = list(parent2.items())
    #print('\no1: ', offspring1, len(offspring1), '\no2: ', offspring2, len(offspring2))

    middle_of_o1, middle_of_o2 =  [], []

    geneA, geneB  = random.randint(0, len_p), random.randint(0, len_p)
    startGene, endGene = min(geneA, geneB), max(geneA, geneB)
    #print(startGene, endGene)
    
    for i in range(0,len_p):
        if i>=startGene and i < endGene:
            middle_of_o1.append(offspring1[i][0])
            middle_of_o2.append(offspring2[i][0])
   
    #print('\n', middle_of_o1, len(middle_of_o1), '\n', middle_of_o2, len(middle_of_o2))
    
    j_o1, j_o2 = 0,0
    
    p1, p2 = list(parent1.items()), list(parent2.items())
    for i in range(len_p):
        if p1[(i+endGene)%len_p][0] not in middle_of_o2:
            offspring2[(j_o2+endGene)%len_p] = p1[(i+endGene)%len_p]
            j_o2+=1
        if p2[(i+endGene)%len_p][0] not in middle_of_o1:
            offspring1[(j_o1+endGene)%len_p] = p2[(i+endGene)%len_p]
            j_o1+=1        

    #print('\n', offspring1, '\n', offspring2)
    return dict(offspring1), dict(offspring2)


def breedPopulation(matingpool):
    '''
    Parameters
    ----------
    matingpool : list
        A list of dictionaries, each dictionaty represents a TSP solution and its id is found in selectionResults. 
        Each key of a dictionaty is  a (city) node and its value is the (city) coordinates

    Returns
    -------
    children : list
        A list of dictionaries, each dictionaty represents a TSP solution and its id is found in selectionResults. 
        Each key of a dictionaty is  a (city) node and its value is the (city) coordinates
        
        Elements of this list are children of individuals in the mating pool. Example:
            if solutions' ids in mating pool are: [9, 7, 0, 6, 5, 0, 1, 6, 2, 3] then:
                ch1, ch2 <- 9, 7
                ...
                ch9, ch10 <- 2,3
            if solutions' ids in mating pool are: [9, 7, 0, 6, 5] then:
                ch1, ch2 <- 9, 7
                ch3, ch4 <- 0,6
                ch5, ch6 <- 9,5 and ch5 is kept arbitrary
    '''
    children = []
    for i in range(1, len(matingpool), 2):
        offspring1, offspring2 = breed(matingpool[i], matingpool[i-1])
        children.extend([offspring1, offspring2])
        
    if len(matingpool)%2:
        offspring1, offspring2 = breed(matingpool[0], matingpool[-1])
        children.append(offspring1) #arbitraty selected
    
    return children


def mutate(individual, rate):
    
    indiv_list = list(individual.items())
    
    for swapped in range(len(indiv_list)):
        if(random.random() < rate):
            swapWith = random.randint(0, len(indiv_list)-1)
            #print('swapped', swapped, 'swapWith', swapWith)
            indiv_list[swapped], indiv_list[swapWith] = indiv_list[swapWith], indiv_list[swapped]

    return dict(indiv_list)


def mutatePopulation(population, mutationRate_population, mutationRate_individual):
    mutatedPop = population
    #print_elements_of_collection('population', population)
    for i in range(0, len(population)):
        if random.random() < mutationRate_population:
            population[i] = mutate(population[i], mutationRate_individual)
            
    return mutatedPop


def nextGeneration(currentGen, eliteSize, mutationRate_population, mutationRate_individual):

    fitness_list = rank_populationElements(currentGen) #current population i.e. population_list
    selected_individuals_list = selection(fitness_list, eliteSize) 
    matingPool_list = create_matingPool(currentGen, selected_individuals_list)#; 
    children = breedPopulation(matingPool_list)
    nextGeneration = mutatePopulation(children, mutationRate_population, mutationRate_individual)    
    return nextGeneration

        
def geneticAlgorithm(nodes_list, popSize, eliteSize, generations, mutationRate_population = 0.5, mutationRate_individual=0.2):
    s =''
    pop = createPopulation(popSize, nodes_list)#; print_elements_of_collection('population_list', population_list)
    print("Initial distance: " + str(1 / rank_populationElements(pop)[0][1]))
    s+= "Initial distance: " + str(1 / rank_populationElements(pop)[0][1]) +'\n'
    
    for i in range(0, generations):
        pop = nextGeneration(pop, eliteSize, mutationRate_population, mutationRate_individual)
    
    print("Final distance: " + str(1 / rank_populationElements(pop)[0][1]))
    s+= "Final distance: " + str(1 / rank_populationElements(pop)[0][1])+'\n'
    
    bestRouteIndex = rank_populationElements(pop)[0][0]
    bestRoute = pop[bestRouteIndex]
    
    return bestRoute, s


def print_elements_of_collection(s, collection, flagElements=False):
    print(s)
    print('collection type: ', type(collection), '|| elements\' type:', type(collection[0]))
    if flagElements:
        for i in range(len(collection)):
            print('element: ',i, '\n', collection[i])
    print('\n\n')    
        

def writeRoute(tspName, route, s_geneAlgo):
    
    f = open(tspName+'_route.txt', 'a')
    s = s_geneAlgo
    f.write(s)
    
    s='Best route: \n'
    for i in range(len(route)):
        s+= str(route[i]+1)+' -> '
        if i % 100 ==0 and i!=0:
            s+='\n'
    s+= str(route[0]+1)+'\n\n'
    #print(s)
    f.write(s)
    
    f.close()
     
    
def main():
        
    tspFileNames = ["ulysses16.tsp","a280.tsp","att532.tsp", "u1817.tsp"]
    #tspFileNames = ["ulysses16.tsp"]

    #tspFileNames=["a280.tsp"]

    intFlag, floatFlag= False, True

    for tspName in tspFileNames:
        
        nodes_dict = {}
        data = read_tsp_data(tspName)
        #dimension = detect_dimension(data)#; print(dimension)
        start_from = detect_start_from(data) #; print(start_from)
        
        nodes_dict = get_nodes_cords(data, start_from, intFlag, floatFlag)#; print(nodes_dict)
        nodes_list = list(nodes_dict.items())#; print_elements_of_collection('nodes_list', nodes_list, True)
        
        popSize = len(nodes_list) * 2 ; 
        eliteSize = len(nodes_list) // 5; generations = int(math.sqrt(len(nodes_list))*25); mutationRate_population, mutationRate_individual = 0.5, 0.2    

        
        bestRoute, s = geneticAlgorithm(nodes_list, popSize, eliteSize, generations)
        #print(bestRoute)
        writeRoute(tspName, list(bestRoute.keys()), s)

        
if __name__ == '__main__':
    main()