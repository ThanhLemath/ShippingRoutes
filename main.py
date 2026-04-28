import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from geopy.distance import geodesic
import geopandas as gpd
from shapely.geometry import Point
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from shapely.geometry import LineString
import xarray as xr
from shapely.geometry import Point
import pickle

# ds = xr.open_dataset(
#     "hrrr.20140730_conus_hrrr.t18z.wrfnatf00.grib2",
#     engine="cfgrib",
#     backend_kwargs={
#         "filter_by_keys": {
#             "typeOfLevel": "heightAboveGround",
#             "level": 10
#         }
#     }
# )


world = gpd.read_file("10m_physical/ne_10m_land.shp")
land = world.geometry.union_all()

def is_on_land(lat, lon):
    return land.contains(Point(lon, lat))

# Vessel Definition 
class Vessel:
    def __init__(self, speed, size, draft, fuel_efficiency):
        self.speed = speed
        self.size = size
        self.draft = draft
        self.fuel_efficiency = fuel_efficiency


vessel = Vessel(speed=15, size=200, draft=10, fuel_efficiency=10)


# Coordinates of Ports (from GPS data) 
port_a_latlon = (35.6762, 139.6503) # Starting point 
port_b_latlon = (33.7405, -118.2626) # Ending point

lat_a, lon_a = port_a_latlon
lat_b, lon_b = port_b_latlon


# Input Scaling Factors 
scaling_lat = 4
scaling_lon = 1.3

# Extended Grid Ranges 
mid_lat = (lat_a + lat_b) / 2
mid_lon = (lon_a + lon_b) / 2
lat_half_range = abs(lat_a - lat_b) * scaling_lat / 2
lon_half_range = abs(lon_a - lon_b) * scaling_lon / 2
lat_min = mid_lat - lat_half_range
lat_max = mid_lat + lat_half_range
lon_min = mid_lon - lon_half_range
lon_max = mid_lon + lon_half_range

if lat_min < -90 or lat_max > 90 or lon_min < -180 or lon_max > 180:
    print("Scaling factors result in invalid grid range. Please adjust scaling factors.")
else:
    try:
        with open("shipping_graph.pkl", "rb") as f:
            graph = pickle.load(f)
        print("Graph loaded successfully.")
        node_positions = {n: graph.nodes[n]['pos'] for n in graph.nodes if 'pos' in graph.nodes[n]}
    except FileNotFoundError:
        print("No saved graph found. Generating from scratch...")

        num_x = 40
        num_y = 40
        time_range = 20
        time_steps = 20
        time_increment = time_range / time_steps
        lat_increment = (lat_max - lat_min) / (num_x - 1)
        lon_increment = (lon_max - lon_min) / (num_y - 1)

        node_positions = {}
        nodes = []

        node_positions["Port_A_0"] = (lat_a, lon_a, 0)
        nodes += ["Port_A_0"]

        
        for k in range(time_steps):
            t = k * time_increment
            node_positions[f"Port_B_{k}"] = (lat_b, lon_b, t)
            nodes += [f"Port_B_{k}"]
            for i in range(num_x):
                for j in range(num_y):
                    node_name = f"Point_{i}_{j}_{k}"
                    x = lat_min + i * lat_increment
                    y = lon_min + j * lon_increment
                
                    node_positions[node_name] = (x, y, t)
                    nodes.append(node_name)

        graph = nx.DiGraph()
        graph.add_nodes_from(nodes)

        moves = [
        (1, 0), (-1, 0), (0, 1), (0, -1),
        (1, 1), (1, -1), (-1, 1), (-1, -1),
        (0, 0)  # stay
        ]
        epsilon_lat = 40
        epsilon_lon = 40
        for k in range(time_steps - 1):
            for i in range(num_x):
                for j in range(num_y):
                    current = f"Point_{i}_{j}_{k}"

                    for di, dj in moves:
                        ni, nj = i + di, j + dj

                        if 0 <= ni < num_x and 0 <= nj < num_y:
                            neighbor = f"Point_{ni}_{nj}_{k+1}"
                            graph.add_edge(current, neighbor)

                        lat_left, lon_left, _ = node_positions[f"Point_0_{j}_{k}"]
                        lat_right, lon_right, _ = node_positions[f"Point_{num_x-1}_{j}_{k}"]

                        if abs(lat_left - lat_right) < epsilon_lat and abs(lon_left - lon_right) < epsilon_lon:
                            graph.add_edge(f"Point_0_{j}_{k}", f"Point_{num_x-1}_{j}_{k+1}")
                            graph.add_edge(f"Point_{num_x-1}_{j}_{k}", f"Point_0_{j}_{k+1}")


        def get_candidate_nodes(port_lat, port_lon, port_t, intcode):
            north = None
            south = None
            east = None
            west = None

            for node, (lat, lon, t) in node_positions.items():
                if node.startswith("Point_"):
                    if intcode == 0:
                        if t == port_t + time_increment:
                            # NORTH (lat greater than port)
                            if lat > port_lat:
                                if north is None or lat < north[1][0]:
                                    north = (node, (lat, lon, t))

                            # SOUTH (lat smaller than port)
                            if lat < port_lat:
                                if south is None or lat > south[1][0]:
                                    south = (node, (lat, lon, t))

                            # EAST (lon greater than port)
                            if lon > port_lon:
                                if east is None or lon < east[1][1]:
                                    east = (node, (lat, lon, t))

                            # WEST (lon smaller than port)
                            if lon < port_lon:
                                if west is None or lon > west[1][1]:
                                    west = (node, (lat, lon, t))
                    if intcode == 1: 
                        if t == port_t: 
                            # NORTH (lat greater than port)
                            if lat > port_lat:
                                if north is None or lat < north[1][0]:
                                    north = (node, (lat, lon, t))

                            # SOUTH (lat smaller than port)
                            if lat < port_lat:
                                if south is None or lat > south[1][0]:
                                    south = (node, (lat, lon, t))

                            # EAST (lon greater than port)
                            if lon > port_lon:
                                if east is None or lon < east[1][1]:
                                    east = (node, (lat, lon, t))

                            # WEST (lon smaller than port)
                            if lon < port_lon:
                                if west is None or lon > west[1][1]:
                                    west = (node, (lat, lon, t))
            
            return [n[0] for n in [north, south, east, west] if n is not None]

        port_a_candidates = get_candidate_nodes(lat_a, lon_a, 0, 0)
        for candidate in port_a_candidates:
            graph.add_edge("Port_A_0", candidate)

        for b_t in range (time_steps - 1):
            port_b_candidates = get_candidate_nodes(lat_b, lon_b, b_t, 1)
            for candidate in port_b_candidates:
                graph.add_edge(candidate, f"Port_B_{b_t + 1}")

        land_nodes = []
        for i in range(num_x):
            for j in range(num_y):
                for k in range(time_steps):
                    node = f"Point_{i}_{j}_{k}"
                    lat = node_positions[f"Point_{i}_{j}_{k}"][0]
                    lon = node_positions[f"Point_{i}_{j}_{k}"][1]
                    if is_on_land(lat, lon):
                        land_nodes.append(node)
                        
        for node in land_nodes:
            graph.remove_node(node)
            node_positions.pop(node)

        def compute_cost(distance_km, wind_speed, wave_height, current_strength, traffic_density, vessel):
            wind_penalty = wind_speed * 0.1
            wave_penalty = wave_height * 2
            current_bonus = current_strength * 0.5
            traffic_penalty = traffic_density * 10
            base_cost = distance_km / vessel.fuel_efficiency
            return base_cost + wind_penalty + wave_penalty + current_bonus + traffic_penalty

        for u, v in graph.edges():
            pos_u = node_positions[u][0:2]
            pos_v = node_positions[v][0:2]
            distance = geodesic(pos_u, pos_v).kilometers

            wind_speed = np.random.uniform(5, 100)
            wave_height = np.random.uniform(2, 100)
            current_strength = np.random.uniform(2, 100)
            traffic_density = np.random.uniform(2, 100)

            cost = compute_cost(
                distance,
                wind_speed,
                wave_height,
                current_strength,
                traffic_density,
                vessel
            )

            graph[u][v].update({
                "weight": cost,
                "distance_km": distance,
                "wind_speed": wind_speed,
                "wave_height": wave_height,
                "current_strength": current_strength,
                "traffic_density": traffic_density
            })
        
        # Create the super-node
        graph.add_node("Final_Destination")

        # Connect all Port_B_k nodes to it
        for k in range(time_steps):
            graph.add_edge(f"Port_B_{k}", "Final_Destination", weight=0)

        for node, pos in node_positions.items():
            graph.nodes[node]['pos'] = pos

        with open("shipping_graph.pkl", "wb") as f: 
            pickle.dump(graph, f)

    # Find shortest path based on cost (weight)
    shortest_path = nx.dijkstra_path(graph, source="Port_A_0", target="Final_Destination", weight='weight')

    # Total cost
    shortest_cost = nx.dijkstra_path_length(graph, source="Port_A_0", target="Final_Destination", weight='weight')
    print("Shortest Path:", shortest_path)
    print("Total Cost:", shortest_cost)

    
    # Plot shortest path nodes
    fig = plt.figure(figsize=(10, 10))
    ax = plt.axes(projection=ccrs.PlateCarree())

    # Set map extent
    padding_lon = 0.05
    padding_lat = 40
    ax.set_extent([
        lon_min - padding_lon, lon_max + padding_lon,
        lat_min - padding_lat, lat_max + padding_lat
    ])

    # Map features
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.LAND, alpha=0.3)
    ax.add_feature(cfeature.OCEAN, alpha=0.3)

    # Extract shortest path coordinates
    path_coords = [node_positions[node] for node in shortest_path[:-1]]
    path_lats = [p[0] for p in path_coords]
    path_lons = [p[1] for p in path_coords]

    # Plot nodes (points)
    ax.scatter(
        path_lons,
        path_lats,
        color='red',
        s=40,
        label='Path Nodes',
        transform=ccrs.PlateCarree()
    )

    # Plot connecting path line
    ax.plot(
        path_lons,
        path_lats,
        color='red',
        linewidth=2,
        label='Optimal Route',
        transform=ccrs.PlateCarree(),
        zorder = 1
    )

    # Highlight start/end
    ax.scatter(lon_a, lat_a, color='green', s=100, label='Start', transform=ccrs.PlateCarree())
    ax.scatter(lon_b, lat_b, color='blue', s=100, label='End', transform=ccrs.PlateCarree())

    plt.legend()
    plt.title("Optimal Route (Shortest Path Only)")
    plt.show()
