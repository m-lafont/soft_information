import soft_information

SI_list = soft_information.parse_scenario('soft_information/data/si1.json')
positions, _ = soft_information.compute_positions(SI_list, nb_points=3)

print(positions.round())