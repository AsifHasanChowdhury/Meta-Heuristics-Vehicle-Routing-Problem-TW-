from EuclidianDistance import calculate_distance


def print_route(route, instance,merge=False):
    route_str = '0'
    sub_route_count = 0
    total_routeCovered =0
   # print(instance["customer_1"]["coordinates"])
    # print("print ",route)
    #print("customer X-cor ", json[])
    for sub_route in route:
        sub_route_distance = 0
        
        sub_route_str = '0'
        
        for customer_id in sub_route:
            # sub_route_distance+= calculate_distance("customer_"+str(customer_id),"customer_"+str(customer_id+1),instance) 
            sub_route_str = f'{sub_route_str} - {customer_id}'
            route_str = f'{route_str} - {customer_id}'
            
        sub_route_str = f'{sub_route_str} - 0'
        for i in range (sub_route_count,len(route)):
               for j in range (0, len(sub_route)-1):
                   #print ("customer ",route[i][j], "customer ",route[i][j+1])
                    sub_route_distance+= calculate_distance("customer_"+str(route[i][j]),"customer_"+str(route[i][j+1]),instance) 

                
               #print("route over ",sub_route_count)
               break
        sub_route_count += 1
        if not merge:
            
            total_routeCovered = total_routeCovered+ sub_route_distance
                    #sub_route_distance+= calculate_distance("customer_"+str(route[i][j]),"customer_"+str(route[i][j+1]),instance) 
            
            print(f'  Vehicle {sub_route_count}\'s route: {sub_route_str} total-area covered {sub_route_distance}')

        route_str = f'{route_str} - 0'
    if merge:
        print(route_str)  
    print("Total Distance Covered by All Vehicles ",total_routeCovered)
