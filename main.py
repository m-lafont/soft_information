import soft_information
import time
pas_temps=1

listepos=[]
while(1):
    SI_list = soft_information.parse_scenario('soft_information/data/si1_2.json')
    positions, _ = soft_information.compute_positions(SI_list, 2, listepos)
    print(positions.round(3))
    listepos+=[positions]
    time.sleep(pas_temps)

