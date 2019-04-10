
# coding: utf-8

# In[15]:

def genRandomGift():
    return(randrange(0,n_gift_type))

def genRandomChild():
    return(randrange(0,n_children))


# In[16]:

def genRandomGift():
    return(randrange(0,n_gift_type))

def genRandomChild():
    return(randrange(0,n_children))

def getAllKeys(dicti , value):
    giftlist = [k for k,v in dicti.items() if v >= value]
    return(giftlist)


# In[17]:

def genGift(ind_class,size):
    
    # fill the giftbucket, each gift with 1000 count.
    giftBucket={}
    for gift in range(0, n_gift_type):     
        giftBucket[gift] = n_gift_quantity                
        
    
    #Generate individual.
    childcount=0
    ind = [-1 for i in range(0,n_children)]      
    child = genRandomChild()
    #print(child)
    while(childcount < size):
        
        if ( ind[child] == -1):             
            if ( (child < triplets)): 
                child = child - (child % 3) 
                giftkeylist = getAllKeys(giftBucket,3)
                if (giftkeylist):
                    gift = random.choice(giftkeylist)                 
                    ind[child] = gift
                    child = child + 1
                    ind[child] = gift
                    child = child + 1
                    ind[child] = gift
                    childcount = childcount + 3
                    giftBucket[gift] = giftBucket[gift] - 3
                    if (giftBucket.get(gift) == 0):
                        del giftBucket[gift]                          
                child = genRandomChild()    
            #allocation of gifts to twins
            elif ( (child >= triplets ) & (child < twin_end )):
                if ((child % 2) == 0):
                    child = child - 1
                giftkeylist = getAllKeys(giftBucket,2) 
                if (giftkeylist):
                    gift = random.choice(giftkeylist)
                    ind[child] = gift
                    child = child + 1
                    ind[child] = gift
                    childcount = childcount + 2
                    giftBucket[gift] = giftBucket[gift] - 2
                    if (giftBucket.get(gift) == 0):
                        del giftBucket[gift]
                child = genRandomChild()    
                               
            else:                 
                giftkeylist = getAllKeys(giftBucket,1) 
                if (giftkeylist):
                    gift = random.choice(giftkeylist)
                    ind[child] = gift 
                    giftBucket[gift] = giftBucket[gift] - 1
                    childcount = childcount + 1
                    if (giftBucket.get(gift) == 0):
                        del giftBucket[gift] 
                child = genRandomChild()        
        else:            
            child = genRandomChild() 
             
    return ind_class(ind)


# In[18]:

from random import randrange
import pandas as pd
import numpy as np
from deap import algorithms, base, creator, tools
from collections import Counter
import utils
import math
import fractions
import random


# In[19]:

n_children = 1000000 # n children to give
n_gift_type = 1000 # n types of gifts available
n_gift_quantity = 1000 # each type of gifts are limited to this quantity
n_gift_pref = 100 # number of gifts a child ranks
n_child_pref = 1000 # number of children a gift ranks
twins = int(math.ceil(0.04 * n_children / 2.) * 2)    # 4% of all population, rounded to the closest number
triplets = int(math.ceil(0.005 * n_children / 3.) * 3 )   # 0.5% of all population, rounded to the closest number
ratio_gift_happiness = 2
ratio_child_happiness = 2
twin_start = triplets
twin_end = triplets + twins


# In[20]:

gift_pref = pd.read_csv('../input/child_wishlist.csv',header=None).drop(0, 1).values
child_pref = pd.read_csv('../input/gift_goodkids.csv',header=None).drop(0, 1).values


# In[21]:

def evalfunc(ind):
    #print("Inside evalFunc ... ind ", len(ind.attr1))
    #print("Inside evalFunc ... ind ",ind.attr1[0:10])
    #print("Inside evalFunc ... ind ",ind.attr1[5000:5010])
    #print("Inside evalFunc ... ind ",ind.attr1[45000:45010])
    
    df = pd.DataFrame(columns=['GiftId','ChildId'])
    df['GiftId'] = [i for i in range(0,n_children)]
    df['ChildId'] = ind.attr1     
    acc = avg_normalized_happiness(df.values.tolist(), child_pref, gift_pref)
    print("The Average Normalized Happiness is ", acc)
    return acc,


# In[22]:

def checking(indiv):
    #print("Inside evalFunc ... ind ", len(ind.attr1))
    #print("Inside evalFunc ... ind ",ind.attr1[0:10])
    #print("Inside evalFunc ... ind ",ind.attr1[5000:5010])
    #print("Inside evalFunc ... ind ",ind.attr1[45000:45010])
    
    df = pd.DataFrame(columns=['GiftId','ChildId'])
    df['GiftId'] = [i for i in range(0,n_children)]
    df['ChildId'] = indiv   
    acc = avg_normalized_happiness(df.values.tolist(), child_pref, gift_pref)
    print("The Average Normalized Happiness is ", acc)
    return acc,


# In[23]:

def lcm(a, b):
    """Compute the lowest common multiple of a and b"""
    # in case of large numbers, using floor division
    #return a * b // math.gcd(a, b)
    return a * b // fractions.gcd(a,b)

def avg_normalized_happiness(pred, child_pref, gift_pref):
    
     
    # check if number of each gift exceeds n_gift_quantity
    gift_counts = Counter(elem[1] for elem in pred)
    for count in gift_counts.values():
        #print "---count--- ",count," -- n_gift_quantity --- ", n_gift_quantity
        assert count <= n_gift_quantity
                
    # check if triplets have the same gift
    for t1 in np.arange(0,int(triplets),3):
        triplet1 = pred[t1]
        triplet2 = pred[t1+1]
        triplet3 = pred[t1+2]
        #print(t1, triplet1, triplet2, triplet3)
        assert triplet1[1] == triplet2[1] and triplet2[1] == triplet3[1]
                
    # check if twins have the same gift
    for t1 in np.arange(triplets,triplets+twins,2):
        twin1 = pred[t1]
        twin2 = pred[t1+1]
        # print(t1)
        assert twin1[1] == twin2[1]
     

    max_child_happiness = n_gift_pref * ratio_child_happiness
    max_gift_happiness = n_child_pref * ratio_gift_happiness
    total_child_happiness = 0
    total_gift_happiness = np.zeros(n_gift_type)
    
    for row in pred:
        child_id = row[0]
        gift_id = row[1]
        
        if ( gift_id < 0 ):            
            print("The child_id and gift_id " + str(child_id) + " " + str(gift_id) )
            
        # check if child_id and gift_id exist
        assert child_id < n_children
        assert gift_id < n_gift_type
        assert child_id >= 0 
        assert gift_id >= 0
        child_happiness = (n_gift_pref - np.where(gift_pref[child_id]==gift_id)[0]) * ratio_child_happiness
        if not child_happiness:
            child_happiness = -1

        gift_happiness = ( n_child_pref - np.where(child_pref[gift_id]==child_id)[0]) * ratio_gift_happiness
        if not gift_happiness:
            gift_happiness = -1

        total_child_happiness += child_happiness
        total_gift_happiness[gift_id] += gift_happiness
    
    print('normalized child happiness=',float(total_child_happiness)/(float(n_children)*float(max_child_happiness)) ,         ', normalized gift happiness',np.mean(total_gift_happiness) / float(max_gift_happiness*n_gift_quantity))

    # to avoid float rounding error
    # find common denominator
    # NOTE: I used this code to experiment different parameters, so it was necessary to get the multiplier
    # Note: You should hard-code the multipler to speed up, now that the parameters are finalized
    denominator1 = n_children*max_child_happiness
    denominator2 = n_gift_quantity*max_gift_happiness*n_gift_type
    common_denom = lcm(denominator1, denominator2)
    multiplier = common_denom / denominator1

    # # usually denom1 > demon2
    return float(math.pow(total_child_happiness*multiplier,3) + math.pow(np.sum(total_gift_happiness),3)) / float(math.pow(common_denom,3))
    # return math.pow(float(total_child_happiness)/(float(n_children)*float(max_child_happiness)),2) + math.pow(np.mean(total_gift_happiness) / float(max_gift_happiness*n_gift_quantity),2)

#random_sub = pd.read_csv('../input/sample_submission_random_v2.csv').values.tolist()
#print(avg_normalized_happiness(random_sub, child_pref, gift_pref))


# In[24]:

def crossOver(ind_class,ind1,ind2):    
    
    indLen = len(ind1.attr1)

    pt1 = randrange(0,n_children)
    pt2 = randrange(0,n_children)
    while(pt1 == pt2):
        pt1 = randrange(0,n_children)
        pt2 = randrange(0,n_children)
     
    if ( pt1 < pt2 ):
        low = pt1
        high = pt2
    else:
        low = pt2
        high = pt1

    modi1 = genModifiedIndividual(ind1.attr1,low,high)
    while ((modi1==None) | (modi1 == ind1)):
        modi1 = genModifiedIndividual(ind1.attr1,low,high)
    
    modi2 = genModifiedIndividual(ind2.attr1,low,high)
    while ((modi2==None) | (modi2 == ind2)):
        modi2 = genModifiedIndividual(ind2.attr1,low,high)
    
    return(ind_class(modi1),ind_class(modi2))
 
   


# In[25]:

def genModifiedIndividual(indiv,low,high):    
    
    indLen = n_children
    
    giftBucket1={}
    for gift in range(0, n_gift_type):     
        giftBucket1[gift] = n_gift_quantity 
    
    #Intialize the individual ind11.
    childcount=0
    ind11 = [-1 for i in range(0,n_children)]
    
    if (low < triplets):
        low = low - ( low % 3 )

    if (high < triplets):
        high  = high  - ( high  % 3 )


    if ( (low >= twin_start) & (low < twin_end)):
        if ((low % 2) == 0):
            low = low - 1

    if ( (high >= twin_start) & (high < twin_end)):
        if ((high % 2) == 0):
            high = high - 1  
            
    #print("After moving --- The low ", low , "and high ", high ,"giftbucket", giftBucket1)
    for i in range(0,low):
        ind11[i] = indiv[i]
        giftBucket1[indiv[i]] = giftBucket1[indiv[i]]-1
        if (giftBucket1.get(indiv[i]) == 0):
            del giftBucket1[indiv[i]]        
    
    for i in range(high,indLen):
        ind11[i] = indiv[i]
        giftBucket1[indiv[i]] = giftBucket1[indiv[i]]-1
        if (giftBucket1.get(indiv[i]) == 0):
            del giftBucket1[indiv[i]]    
    
    rangeValue = high - low
    #print("The rangeValue ", rangeValue)
    i=0
    child = randrange(low,high)
    while (i < rangeValue):              
        #print("The value of i ",i,"and child ", child)
        if ( ind11[child] == -1): 
            
                if ( (child < triplets)):
                    
                    if ( (child % 3) != 0):  
                        child = child - (child % 3) 
                    giftkeylist = getAllKeys(giftBucket1,3)
                    if (giftkeylist):
                        gift = random.choice(giftkeylist)                         
                    else:
                        return None
                    #print(giftBucket1, "the gift is ", gift,"The individual ", ind11,"The child is ", child,"giftbucket", giftBucket1)
                    ind11[child] = gift
                    child = child + 1
                    ind11[child] = gift
                    child = child + 1
                    ind11[child] = gift                         
                    i = i + 3
                    giftBucket1[gift] = giftBucket1[gift] - 3
                    if (giftBucket1.get(gift) == 0):
                        del giftBucket1[gift]                        
                    child = randrange(low,high)            
                elif ( (child >= triplets) & (child < twin_end)):  
                    
                        if ( (child % 2) != 0):
                            child = child -1
                        giftkeylist = getAllKeys(giftBucket1,2)
                        if (giftkeylist):
                            gift = random.choice(giftkeylist)                         
                        else:
                            return None 
                                                     
                        #print(giftBucket1, "the gift is ", gift,"The individual ", ind11,"The child is ", child,"giftbucket", giftBucket1)
                        ind11[child] = gift
                        child = child + 1
                        ind11[child] = gift
                            #childcount = childcount + 2
                        i = i + 2
                        giftBucket1[gift] = giftBucket1[gift] - 2
                        if (giftBucket1.get(gift) == 0):
                            del giftBucket1[gift]
                        child = randrange(low,high)                   
                else:  
                    
                    giftkeylist = getAllKeys(giftBucket1,1)
                    if (giftkeylist):
                        gift = random.choice(giftkeylist)                         
                    else:
                        return None                                                               
                    ind11[child] = gift 
                    giftBucket1[gift] = giftBucket1[gift] - 1
                    #childcount = childcount + 1
                    i = i + 1
                    if (giftBucket1.get(gift) == 0):
                        del giftBucket1[gift] 
                    child = randrange(low,high)      
        else:                 
            child = randrange(low,high)
            
    return(ind11)


#  
# def genModifiedIndividual_new(indiv,low,high):    
#     
#     indLen = n_children
#     
#     giftBucket1={}
#     for gift in range(0, n_gift_type):     
#         giftBucket1[gift] = n_gift_quantity 
#     
#     #Intialize the individual ind11.
#     childcount=0
#     ind11 = [-1 for i in range(0,n_children)]
#     
#     if (low < triplets):
#         low = low - ( low % 3 )
#         
#     if (high < triplets):
#         high  = high  - ( high  % 3 )
#          
#     
#     if ( (low >= twin_start) & (low < twin_end)):
#         if ((low % 2) == 0):
#             low = low - 1
#             
#     if ( (high >= twin_start) & (high < twin_end)):
#          if ((high % 2) == 0):
#             high = high - 1
#               
#             
#     #print("After moving --- The low ", low , "and high ", high ,"giftbucket", giftBucket1)
#     for i in range(0,low):
#         ind11[i] = indiv[i]
#         giftBucket1[indiv[i]] = giftBucket1[indiv[i]]-1
#         if (giftBucket1.get(indiv[i]) == 0):
#             del giftBucket1[indiv[i]]        
#     
#     for i in range(high,n_children):
#         ind11[i] = indiv[i]
#         giftBucket1[indiv[i]] = giftBucket1[indiv[i]]-1
#         if (giftBucket1.get(indiv[i]) == 0):
#             del giftBucket1[indiv[i]]    
#    
# 
#     rangeValue = high - low   
#     rangeValue = rangeValue + 1
#     i=0
#     child = randrange(low,high)
#     while (i < rangeValue):       
#         #print("The value of i ",i,"and child ", child)
#         if ( ind11[child] == -1):                 
#                 if ( (child < triplets )):                 
#                     if ( (child % 3) != 0):  
#                         child = child - (child % 3) 
#                     giftkeylist = getAllKeys(giftBucket1,3)
#                     if (giftkeylist):
#                         gift = random.choice(giftkeylist)                         
#                     else:
#                         return None
#                     #print(giftBucket1, "the gift is ", gift,"The individual ", ind11,"The child is ", child,"giftbucket", giftBucket1)
#                     ind11[child] = gift
#                     child = child + 1
#                     ind11[child] = gift
#                     child = child + 1
#                     ind11[child] = gift                         
#                     i = i + 3
#                     giftBucket1[gift] = giftBucket1[gift] - 3
#                     if (giftBucket1.get(gift) == 0):
#                         del giftBucket1[gift]                        
#                     child = randrange(low,high)                                           
#                 elif ( (child >= twin_start ) & (child < twin_end)):                     
#                     if ((child % 2) == 0):
#                         child = child - 1
#                     giftkeylist = getAllKeys(giftBucket1,2) 
#                     if (giftkeylist):
#                         gift = random.choice(giftkeylist)                       
#                     else:
#                         return None
#                     ind11[child] = gift
#                     child = child + 1
#                     ind11[child] = gift                        
#                     i = i + 2
#                     giftBucket1[gift] = giftBucket1[gift] - 2
#                     if (giftBucket1.get(gift) == 0):
#                         del giftBucket1[gift]
#                     child = randrange(low,high)                   
#                 else:  
#                     giftkeylist = getAllKeys(giftBucket1,1) 
#                     if (giftkeylist):
#                         gift = random.choice(giftkeylist)
#                     else:
#                         return None
#                     ind11[child] = gift 
#                     giftBucket1[gift] = giftBucket1[gift] - 1                                        
#                     i = i + 1
#                     if (giftBucket1.get(gift) == 0):
#                         del giftBucket1[gift] 
#                     child = randrange(low,high)      
#         else:
#             child = randrange(low,high)
#             
#     return(ind11)
#     

# In[26]:

def mutation(ind1):    
    
    mutated = performMutation(ind1.attr1[:])
    while (mutated == ind1.attr1):
        mutated = performMutation(ind1.attr1[:])
        
    return(ind1,)


def performMutation(indiv):
    
    mutPoint = randrange(0,n_children)
    
    if (mutPoint < triplets):
                
        mutPoint = mutPoint -(mutPoint % 3)

        subPt = randrange(0,triplets)
        while (subPt == mutPoint):
            subPt = randrange(0,triplets)
            subPt = subPt - (subPt % 3)                          
    
        temp1 = indiv[subPt]
        temp2 = indiv[subPt+1]
        temp3 = indiv[subPt+2]
        indiv[subPt] = indiv[mutPoint]
        indiv[subPt+1] = indiv[mutPoint+1]
        indiv[subPt+2] = indiv[mutPoint+2]
        indiv[mutPoint] = temp1
        indiv[mutPoint+1] = temp2
        indiv[mutPoint+2] = temp3        
     
    elif  (mutPoint >= twin_start) & (mutPoint < twin_end ):
                
        mutPoint = mutPoint -(mutPoint % 2)

        subPt = randrange(twin_start,twin_end)
        while (subPt == mutPoint):
            subPt = randrange(twin_start,twin_end)
            subPt = subPt - (subPt % 2)                          
    
        temp1 = indiv[subPt]
        temp2 = indiv[subPt+1]
        indiv[subPt] = indiv[mutPoint]
        indiv[subPt+1] = indiv[mutPoint+1]
        indiv[mutPoint] = temp1
        indiv[mutPoint+1] = temp2

    else:
        subPt = randrange(twins,n_children)
        temp1 = indiv[subPt]
        indiv[subPt] = indiv[mutPoint]
        indiv[mutPoint] = temp1

    return(indiv)


# In[27]:

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual",utils.MyContainer, fitness=creator.FitnessMax)

toolbox = base.Toolbox()


# In[28]:

toolbox.register("individual", genGift,  creator.Individual, size=n_children)
#toolbox.register("select", tools.selNSGA2)
toolbox.register("select",tools.selTournament, tournsize=3)


# In[29]:

# Make the population
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
hof = tools.HallOfFame(1)
toolbox.register("evaluate", evalfunc)
toolbox.register("mate", crossOver,creator.Individual)
toolbox.register("mutate", mutation)

POP_SIZE = 50
MU = POP_SIZE # The number of individuals to select for the next generation
LAMBDA = POP_SIZE #The number of children to produce at each generation 
MAX_GEN = 5
MUT_PROB = 0.02
CX_PROB = 0.5

pop = toolbox.population(n=POP_SIZE)
result, log = algorithms.eaMuPlusLambda(pop, toolbox, MU, LAMBDA, CX_PROB, MUT_PROB,
                                        ngen=MAX_GEN,halloffame=hof,verbose= True)


# In[30]:

fronts = tools.emo.sortLogNondominated(result, len(result))


# In[31]:

best = tools.selBest(pop, k=1)


# In[32]:

#best[0].attr1 --> This gives the best individual


# In[33]:

checking(best[0].attr1)


# ind = toolbox.individual()
# 
# evalfunc(ind)
# 
# toolbox.population(n=2)
# 
# n_children = 10 # n children to give
# n_gift_type = 5 # n types of gifts available
# n_gift_quantity = 2 # each type of gifts are limited to this quantity
# n_gift_pref = 10 # number of gifts a child ranks
# n_child_pref = 1000 # number of children a gift ranks
# twins = int(.4 * n_children)    # 0.4% of all population, rounded to the closest even number
# ratio_gift_happiness = 2
# ratio_child_happiness = 2
# genGift(10)
