
#euclidian distance calculation
def calculate_distance(customer1, customer2,instance):
    if(customer1 == "customer_0"):

        return ((40 - instance[customer2]['coordinates']['x']**2 + \
                50 - instance[customer2]['coordinates']['y'])**2)**0.5
    # print(customer1," ",instance[customer1]['coordinates']['x'] , instance[customer1]['coordinates']['y'])
    # print(customer2," ",instance[customer2]['coordinates']['x'] , instance[customer2]['coordinates']['y'])

    return ((instance[customer1]['coordinates']['x'] - instance[customer2]['coordinates']['x'])**2 + \
        (instance[customer1]['coordinates']['y'] - instance[customer2]['coordinates']['y'])**2)**0.5
