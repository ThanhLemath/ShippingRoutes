import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from geopy.distance import geodesic
import geopandas as gpd
from shapely.geometry import Point
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import xarray as xr
from shapely.geometry import Point
import pickle


ds_list = []
for i in range(16):
    f = f"data/hrrr.t18z.wrfnatf{i:02d}.grib2"
    ds_list.append(
        xr.open_dataset(
            f,
            engine="cfgrib",
            backend_kwargs={
                "filter_by_keys": {
                    "typeOfLevel": "heightAboveGround",
                    "level": 10
                }
            }
        )
    )

lat_ref = ds_list[0]["latitude"].values
lon_ref = ds_list[0]["longitude"].values
u_cache   = [ds["u10"].values for ds in ds_list]
v_cache   = [ds["v10"].values for ds in ds_list]

# for i, ds in enumerate(ds_list[1:], start=1):
#     lat_i = ds["latitude"].values
#     lon_i = ds["longitude"].values

#     if not (np.allclose(lat_ref, lat_i, atol=1e-6) and
#             np.allclose(lon_ref, lon_i, atol=1e-6)):
#         raise ValueError(f"Grid mismatch at timestep {i}")

def find_initial_index(lat0, lon0):
    dist2 = (lat_ref - lat0)**2 + ((lon_ref - lon0) * np.cos(np.radians(lat0)))**2
    return np.unravel_index(np.argmin(dist2), lat_ref.shape)

def find_local_index(lat0, lon0, y_prev, x_prev, window=10):
    y_min = max(0, y_prev - window)
    y_max = min(lat_ref.shape[0], y_prev + window + 1)

    x_min = max(0, x_prev - window)
    x_max = min(lat_ref.shape[1], x_prev + window + 1)

    lat_sub = lat_ref[y_min:y_max, x_min:x_max]
    lon_sub = lon_ref[y_min:y_max, x_min:x_max]

    dist2 = (lat_sub - lat0)**2 + ((lon_sub - lon0) * np.cos(np.radians(lat0)))**2

    dy, dx = np.unravel_index(np.argmin(dist2), dist2.shape)

    return y_min + dy, x_min + dx

def GetWind(time, y, x):
    u = u_cache[time][y, x]
    v = v_cache[time][y, x]
    speed = np.sqrt(u*u + v*v)
    return u, v, speed


world = gpd.read_file("10m_physical/ne_10m_land.shp")
land = world.geometry.union_all()

def is_on_land(lat, lon):
    return land.contains(Point(lon, lat))



# Vessel Definition 
class Vessel:
    def __init__(self, size, draft, fuel_efficiency):
        self.size = size
        self.draft = draft
        self.fuel_efficiency = fuel_efficiency


vessel = Vessel(size=200, draft=10, fuel_efficiency=10)


# Coordinates of Ports (from GPS data) 
port_a_latlon = (40.6688, -74.0451)  # Santa Monica / LA coast
port_b_latlon = (25.7781, -80.1794)  # Rockaway Beach / Long Island coast

lat_a, lon_a = port_a_latlon
lat_b, lon_b = port_b_latlon


# Input Scaling Factors 
scaling_lat = 5
scaling_lon = 1

# Extended Grid Ranges 
mid_lat = (lat_a + lat_b) / 2
mid_lon = (lon_a + lon_b) / 2
lat_half_range = abs(lat_a - lat_b) * scaling_lat / 2
lon_half_range = abs(lon_a - lon_b) * scaling_lon / 2
lat_min = mid_lat - lat_half_range
lat_max = mid_lat + lat_half_range
lon_min = mid_lon - lon_half_range
lon_max = mid_lon + lon_half_range

num_lat = 15
num_lon = 15
time_range = 16
time_steps = 16
time_increment = time_range / time_steps
lat_increment = (lat_max - lat_min) / (num_lat - 1)
lon_increment = (lon_max - lon_min) / (num_lon - 1)


index_map = {}
for i in range(num_lat):
    for j in range(num_lon):
        if i==0 and j==0:
            index_map[(0, 0)] = find_initial_index(lat_min, lon_min)

        elif i == 0 and j != 0:
            index_map[(i, j)] = find_local_index(lat_min + i * lat_increment, 
                                                 lon_min + j * lon_increment, 
                                                 index_map[(i, j - 1)][0],
                                                 index_map[(i, j - 1)][1])
        
        elif i != 0:
            index_map[(i, j)] = find_local_index(lat_min + i * lat_increment, 
                                                 lon_min + j * lon_increment, 
                                                 index_map[(i - 1, j)][0],
                                                 index_map[(i - 1, j)][1])

y_b, x_b = find_initial_index(lat_b, lon_b)


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

        node_positions = {}
        nodes = []

        node_positions["Port_A_0"] = (lat_a, lon_a, 0)
        nodes += ["Port_A_0"]

        land_mask = {}
        for i in range(num_lat):
            for j in range(num_lon):
                lat = lat_min + i * lat_increment
                lon = lon_min + j * lon_increment
                land_mask[(i, j)] = is_on_land(lat, lon)

        
        for k in range(time_steps):
            t = k * time_increment
            node_positions[f"Port_B_{k}"] = (lat_b, lon_b, t)
            nodes += [f"Port_B_{k}"]

            for i in range(num_lat):
                for j in range(num_lon):
                    node_name = f"Point_{i}_{j}_{k}"
                    lat = lat_min + i * lat_increment
                    lon = lon_min + j * lon_increment

                    if land_mask[(i, j)]:
                        continue
                                
                    node_positions[node_name] = (lat, lon, t)
                    nodes.append(node_name)

        graph = nx.DiGraph()
        graph.add_nodes_from(nodes)
        del nodes

        moves = [
        (1, 0), (-1, 0), (0, 1), (0, -1),
        (1, 1), (1, -1), (-1, 1), (-1, -1)
        ]

        for k in range(time_steps - 1):
            for i in range(num_lat):
                for j in range(num_lon):

                    if f"Point_{i}_{j}_{k}" not in graph:
                        continue
                    current = f"Point_{i}_{j}_{k}"

                    for di, dj in moves:
                        ni, nj = i + di, j + dj

                        if 0 <= ni < num_lat and 0 <= nj < num_lon:                            
                            if f"Point_{ni}_{nj}_{k+1}" not in graph:
                                continue

                            neighbor = f"Point_{ni}_{nj}_{k+1}"
                            graph.add_edge(current, neighbor)

                        
        epsilon_lon = 1
        for k in range(time_steps - 1):
                for i in range(num_lat): 
                    if f"Point_{i}_{0}_{k}" not in graph or f"Point_{i}_{num_lon - 1}_{k}" not in graph:
                        continue

                    lat_left, lon_left, _ = node_positions[f"Point_{i}_{0}_{k}"]
                    lat_right, lon_right, _ = node_positions[f"Point_{i}_{num_lon - 1}_{k}"]

                    if abs(lon_left - lon_right) < epsilon_lon:
                        graph.add_edge(f"Point_{i}_{0}_{k}", f"Point_{i}_{num_lon - 1}_{k+1}")
                        graph.add_edge(f"Point_{i}_{num_lon - 1}_{k}", f"Point_{i}_{0}_{k+1}")
        

        def get_candidate_nodes(port_lat, port_lon, port_t, intcode):
            candidates = []

            for node, (lat, lon, t) in node_positions.items():
                if not node.startswith("Point_"):
                    continue

                # Time filtering
                if intcode == 0 and t != port_t + time_increment:
                    continue
                if intcode == 1 and t != port_t:
                    continue

                # Compute distance 
                dist = geodesic((port_lat, port_lon),(lat, lon)).kilometers

                candidates.append((dist, node))

            # Sort by distance
            candidates.sort(key=lambda x: x[0]) # Choose n

            # Return the n closest nodes
            return [node for _, node in candidates[:2]] 

        port_a_candidates = get_candidate_nodes(lat_a, lon_a, 0, 0)
        for candidate in port_a_candidates:
            graph.add_edge("Port_A_0", candidate)

        port_b_candidates = get_candidate_nodes(lat_b, lon_b, 0, 1)

        pairs = set()
        for node in port_b_candidates:
            _, i, j, k = node.split("_")
            pairs.add((int(i), int(j)))

        for i, j in pairs:
            for k in range(time_steps - 1):
                graph.add_edge(f"Point_{i}_{j}_{k}", f"Port_B_{k + 1}")


        print("Port_A_0 present:", "Port_A_0" in graph.nodes)

        for k in range(time_steps):
            print(f"Port_B_{k} present:", f"Port_B_{k}" in graph.nodes)


        def compute_cost(distance_km,
                travel_vec,
                wind_speed,
                wind_vec,
                wave_height,
                current_strength,
                traffic_density,
                vessel):
            
            wind_penalty = wind_speed * 0.1
            wave_penalty = wave_height * 1
            current_bonus = current_strength * 0.5
            traffic_penalty = traffic_density * 1
            base_cost = distance_km / vessel.fuel_efficiency
            if np.linalg.norm(travel_vec) < 1e-9:
                wind_cost = 0.0
            else:
                travel_unit = travel_vec / np.linalg.norm(travel_vec)
                wind_alignment = np.dot(travel_unit, wind_vec)
                wind_cost = 150 - 100 * wind_alignment

            return base_cost + wind_penalty + wave_penalty + current_bonus + traffic_penalty + wind_cost

        for u, v in graph.edges():

            if u == "Final_Destination" or v == "Final_Destination":
                continue

            if v.startswith("Point_"):
                _, i, j, k = v.split("_")
                i = int(i)
                j = int(j)
                k = int(k)
                y, x = index_map[(i, j)]
            

            elif v.startswith("Port_B"):
                y, x = y_b, x_b
                _, _, k = v.split("_")
                k = int(k)

            pos_u = node_positions[u][0:2]
            pos_v = node_positions[v][0:2]
            lat_u, lon_u = pos_u
            lat_v, lon_v = pos_v
            distance = geodesic(pos_u, pos_v).kilometers
            travel_vec = np.array(pos_v) - np.array(pos_u)
            travel_vec[0] = travel_vec[0] *  111320
            lat_mid = 0.5 * (lat_u + lat_v)
            travel_vec[1] = (lon_v - lon_u) * 111320 * np.cos(np.radians(lat_mid))
            wy, wx, wind_speed = GetWind(k, y, x)

            wind_vec = np.array([wx, wy])
            norm = np.linalg.norm(wind_vec)

            if norm > 0:
                wind_vec /= norm
            else:
                wind_vec = np.array([0.0, 0.0])

            wave_height = np.random.uniform(2, 2)
            current_strength = np.random.uniform(2, 2)
            traffic_density = np.random.uniform(2, 2)

            cost = compute_cost(
                distance,
                travel_vec,
                wind_speed,
                wind_vec,
                wave_height,
                current_strength,
                traffic_density,
                vessel
            )

            graph[u][v].update({
                "weight": cost,
                "distance": distance,
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

        # Plot every node
    fig = plt.figure(figsize=(10, 10))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.stock_img()
    ax.set_ylim(lat_min, lat_max)


    # Extract coordinates
    lats = [pos[0] for pos in node_positions.values()]
    lons = [pos[1] for pos in node_positions.values()]

    # Plot nodes (points)
    ax.scatter(
        lons,
        lats,
        color='red',
        s=1,
        label='Path Nodes',
        transform=ccrs.PlateCarree()
    )

    
    lons = []
    lats = []

    def fix_dateline(lon1, lat1, lon2, lat2):
        # detect wrap
        if abs(lon1 - lon2) > 180:
            if lon1 < 0:
                lon1 += 360
            if lon2 < 0:
                lon2 += 360
        return lon1, lat1, lon2, lat2


    for u, v in graph.edges():

        if u == "Final_Destination" or v == "Final_Destination":
            continue

        lat_u, lon_u, _ = node_positions[u]
        lat_v, lon_v, _ = node_positions[v]

        lon_u, lat_u, lon_v, lat_v = fix_dateline(lon_u, lat_u, lon_v, lat_v)

        lons += [lon_u, lon_v, None]
        lats += [lat_u, lat_v, None]


    ax.plot(
        lons,
        lats,
        color="gray",
        linewidth=0.3,
        alpha=0.4,
        transform=ccrs.PlateCarree(),
        zorder = 2
    )

    plt.legend()
    plt.title("Nodes left")
    plt.show()

    # Find shortest path based on cost (weight)
    shortest_path = nx.dijkstra_path(graph, source="Port_A_0", target="Final_Destination", weight='weight')

    # Total cost
    shortest_cost = nx.dijkstra_path_length(graph, source="Port_A_0", target="Final_Destination", weight='weight')
    print("Shortest Path:", shortest_path)
    print("Total Cost:", shortest_cost)

    # Plot shortest path nodes
    fig = plt.figure(figsize=(10, 10))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.stock_img()
    ax.set_ylim(lat_min, lat_max)


    # Extract shortest path coordinates
    path_lats = [node_positions[node][0] for node in shortest_path[:-1]]
    path_lons = [node_positions[node][1] for node in shortest_path[:-1]]

    # Plot nodes (points)
    ax.scatter(
        path_lons,
        path_lats,
        color='red',
        s=1,
        label='Path Nodes',
        transform=ccrs.PlateCarree()
    )

    
    lons = []
    lats = []

    for i in range(len(shortest_path[:-2])):
        u = shortest_path[i]
        v = shortest_path[i+1]

        lat_u, lon_u, _ = node_positions[u]
        lat_v, lon_v, _ = node_positions[v]

        lon_u, lat_u, lon_v, lat_v = fix_dateline(lon_u, lat_u, lon_v, lat_v)

        lons += [lon_u, lon_v, None]
        lats += [lat_u, lat_v, None]


    # Plot the actual optimal path in red
    ax.plot(
    lons,
    lats,
    color='red',
    linewidth=1,
    label='Optimal Route',
    transform=ccrs.PlateCarree(),
    zorder=5
    )

    plt.legend()
    plt.title("Nodes left")
    plt.show()
