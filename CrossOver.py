import random

def cx_partially_mapped(ind1, ind2):

    child1 = [0]*len(ind1)
    child2 = [0] *len(ind2)
    # print(child1)
    cxpoint1, cxpoint2 = sorted(random.sample(range(min(len(ind1), len(ind2))), 2))
    print("CXPOINT 1",cxpoint1)
    print("CXPOINT 2",cxpoint2)
    backup = cxpoint1
    part1 = ind1[cxpoint1:cxpoint2+1] #slice data 1
    part2 = ind2[cxpoint1:cxpoint2+1] #slice data 2
    rule1to2 = list(zip(part1, part2))
    i=0
    while(cxpoint1<cxpoint2):
        child1[cxpoint1]=part2[i]
        cxpoint1+=1 # 6
        i+=1
    i=0
    cxpoint1 = backup

    while(cxpoint1<cxpoint2):
        child2[cxpoint1]=part1[i]
        cxpoint1+=1
        i+=1
    i=0


    for t1,t2 in rule1to2:
        print(f't1 {t1} t2 {t2}')


    #print("before cross ",child1)
    while i<len(child1):
        
        if(child1[i]==0):
            if(ind1[i] not in child1):
                child1[i]= ind1[i]
            else:    
                for t1,t2 in rule1to2:
                    if (t1==child1[i]):
                        if(t2 not in child1):
                            print("fairuz bolse 0 ", t2)
                            child1[i] = t2
                        else:
                            t1 =t2
        i+=1
           
    #print("CROSS ",child1)
    return child1, child2


def order_cross_over(ind1, ind2):
    
    # print("Crossing")
    child1 = [0]*len(ind1)
    child2 = [0] *len(ind2)
    
    cxpoint1, cxpoint2 = sorted(random.sample(range(min(len(ind1), len(ind2))), 2))


    backup = cxpoint1
    part1 = ind1[cxpoint1:cxpoint2+1] #slice data 1
    part2 = ind2[cxpoint1:cxpoint2+1] #slice data 2

    i=0
    while(cxpoint1<cxpoint2):
        child1[cxpoint1]=part1[i]
        cxpoint1+=1 
        i+=1
    i=0
    cxpoint1 = backup
  
    return child1, child2


# parent1 = [1, 2, 3, 4, 5, 6, 7, 8, 9]
# parent2 = [3, 9, 4, 7, 6, 5, 2, 1, 8]

#print(cx_partially_mapped([1,2,3,4,5,6,7,8,9],[3,9,4,7,6,5,2,1,8]))



def cx_partially(ind1, ind2):

    child1 = [-1]*len(ind1)
    child2 = [-1] *len(ind2)
    # print(child1)
    cxpoint1, cxpoint2 = sorted(random.sample(range(min(len(ind1), len(ind2))), 2))

    
    if(cxpoint1>cxpoint2):
        temp = cxpoint1
        cxpoint1 = cxpoint2
        cxpoint2 = temp

  
    i=cxpoint1
    while (i<cxpoint2):
        child2[i] = ind1[i]
        i+=1
    
    j=cxpoint1
    while (j<cxpoint2):
        child1[j] = ind2[j]
        j+=1

    #make child 1 
    childIndex = 0
    while(childIndex<len(child1)):

        if(child1[childIndex]== -1):
            indexJumper = childIndex
            parent1Check = False
            parent2Check = False
            while(True):

                if(ind1[indexJumper] in child1 and parent1Check==False):
                    indexJumper = ind1.index(ind1[indexJumper])
                    parent1Check = True
                    parent2Check = False
                
                if(ind2[indexJumper] in child1 and parent2Check==False):
                    indexJumper = ind2.index(ind1[indexJumper])
                    parent1Check = False
                    parent2Check = True

                if(ind1[indexJumper] not in child1):
                    child1[childIndex] = ind1[indexJumper]
                    break

                if(ind2[indexJumper] not in child1):
                    child1[childIndex] = ind2[indexJumper]
                    break
                    

        childIndex+=1

    #make child 2
    childIndex = 0
    while(childIndex<len(child2)):

        if(child2[childIndex]== -1):
            indexJumper = childIndex
            parent1Check = False
            parent2Check = False
            while(True):

                if(ind1[indexJumper] in child2 and parent1Check==False):
                    indexJumper = ind1.index(ind1[indexJumper])
                    parent1Check = True
                    parent2Check = False
                
                if(ind2[indexJumper] in child2 and parent2Check==False):
                    indexJumper = ind2.index(ind1[indexJumper])
                    parent1Check = False
                    parent2Check = True

                if(ind1[indexJumper] not in child2):
                    child2[childIndex] = ind1[indexJumper]
                    break

                if(ind2[indexJumper] not in child2):
                    child2[childIndex] = ind2[indexJumper]
                    break
                    

        childIndex+=1


          
        


    return child1, child2
    # part1 = ind1[cxpoint1:cxpoint2] #slice data 1
    # part2 = ind2[cxpoint1:cxpoint2] #slice data 2
    # print(child1)
    # print(child2)

    #return child1,child1
# print(cx_partially([1,2,3,4,5,6,7,8,9],[3,9,4,7,6,5,2,1,8]))

#print(cx_partially_mapped([1,2,3,4,5,6,7,8,9],[3,9,4,7,6,5,2,1,8]))