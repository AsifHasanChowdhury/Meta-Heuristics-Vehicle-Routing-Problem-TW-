def elitistSelection(List1,List2):
    sorted_pairs = sorted(zip(List2, List1), key=lambda x: x[0],reverse=True)
    List2, List1 = zip(*sorted_pairs)

    return list(List1)

    
# print()

List1 = [[2, 4, 6, 8], [1, 4, 2, 6], [1, 2], [5, 6]]
List2 = [1, 10, 6, 14]

print(elitistSelection(List1,List2))