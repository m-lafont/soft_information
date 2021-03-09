import soft_information
import time
pas_temps=10

listepos=[]
listepos_temps=[]
while(1):
    SI_list = soft_information.parse_scenario('soft_information/data/si1_2.json')
    positions, _ = soft_information.compute_positions(SI_list, 2, listepos)
    T= soft_information.get_time()
#    print(positions.round(3))
    listepos+=[positions.round(3)]
    listepos_temps.append([T[-1],positions.round(3)])
    print(listepos_temps[-1])
    time.sleep(pas_temps)
