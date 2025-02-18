from RouteEncoding import individual_to_route_decoding
count = 0

def evaluate_individual(individual, instance, unit_cost=1.0, init_cost=0, wait_cost=0, delay_cost=0):
   
    global count
    #print(f"Individual {count}")
    count+=1
    total_cost = 0
    route = individual_to_route_decoding(individual, instance)
    total_cost = 0
    sub_route_distance = 0
    #print(f"Route   {route}")
    for sub_route in route:
        sub_route_time_cost = 0
        elapsed_time = 0
        last_customer_id = 0
        for customer_id in sub_route:
            # Calculate section distance
            distance = instance['distance_matrix'][last_customer_id][customer_id]
            # Update sub-route distance
            sub_route_distance = sub_route_distance + distance
            # Calculate time cost
            arrival_time = elapsed_time + distance
             # ready time is the starting time of customer. 3pm . Suppose arrival time 3:5 pm
              # due time is last time of customer. 3:10 pm. Suppose arrival time 3:5 pm
            time_cost = wait_cost * max(instance[f'customer_{customer_id}']['ready_time'] - arrival_time, 0) + delay_cost * max(arrival_time - instance[f'customer_{customer_id}']['due_time'], 0)
            # Update sub-route time cost
            sub_route_time_cost = sub_route_time_cost + time_cost
            # Update elapsed time
            elapsed_time = arrival_time + instance[f'customer_{customer_id}']['service_time']
            # Update last customer ID
            last_customer_id = customer_id
        # Calculate transport cost
        # sub_route_distance = sub_route_distance + instance['distance_matrix'][last_customer_id][0]
        sub_route_transport_cost = init_cost + unit_cost * sub_route_distance
        # Obtain sub-route cost
        sub_route_cost = sub_route_time_cost + sub_route_transport_cost
        # Update total cost
        total_cost = total_cost + sub_route_cost 
    # fitness = 1.0 / total_cost
    fitness = 1/sub_route_distance
    # print("Helloooo",fitness)
    # return (fitness,)
    
    return fitness
 