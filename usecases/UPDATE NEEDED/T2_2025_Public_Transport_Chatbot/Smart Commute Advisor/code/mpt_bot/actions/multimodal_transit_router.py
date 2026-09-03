import pandas as pd
import os
from datetime import datetime
import logging
import heapq
from collections import defaultdict
from geopy.geocoders import Nominatim
from geopy.distance import geodesic
import numpy as np
from scipy.spatial import cKDTree
import openrouteservice
import folium
import time

class MultimodalTransitRouter:
    def __init__(self):
        self.logger = logging.getLogger('actions.actions')
        self.api_key = '5b3ce3597851110001cf6248a6b7c97bb850491794bb504b30e2f2f7'
        start_time = time.time()
        self.logger.info("Starting router initialization")
        
        self.load_transit_data()
        self._build_shape_cache()
        
        self.logger.info(f"Total initialization took {time.time() - start_time:.2f} seconds")

    def load_transit_data(self):
        """Load transit data"""
        bus_dir = os.path.join('..', 'mpt_data', '4')
        train_dir = os.path.join('..', 'mpt_data', '2')
        
        # Load stops and convert to numpy arrays
        bus_stops = pd.read_csv(os.path.join(bus_dir, 'stops.txt'))
        bus_stops['mode'] = '4'
        train_stops = pd.read_csv(os.path.join(train_dir, 'stops.txt'))
        train_stops['mode'] = '2'
        
        self.stops_df = pd.concat([bus_stops, train_stops], ignore_index=True)
        
        self.stop_coords = np.array(self.stops_df[['stop_lat', 'stop_lon']].values, dtype=float)
        self.stop_ids = np.array(self.stops_df['stop_id'].astype(str))
        
        # Create quick lookup dictionaries
        self.stop_idx_lookup = {str(stop_id): idx for idx, stop_id in enumerate(self.stop_ids)}
        self.stop_info = self.stops_df.to_dict('records')
        self.stop_info_lookup = {str(stop['stop_id']): stop for stop in self.stop_info}
        
        # Load routes with minimal columns
        routes_cols = ['route_id', 'route_short_name']
        bus_routes = pd.read_csv(os.path.join(bus_dir, 'routes.txt'))[routes_cols]
        bus_routes['mode'] = '4'
        train_routes = pd.read_csv(os.path.join(train_dir, 'routes.txt'))[routes_cols]
        train_routes['mode'] = '2'
        
        self.routes = pd.concat([bus_routes, train_routes], ignore_index=True)
        
        # Create route lookup
        self.route_info = {
            str(row['route_id']): row 
            for _, row in self.routes.iterrows()
        }
        
        # Load and process stop times efficiently
        stop_times_cols = ['trip_id', 'stop_id', 'stop_sequence']
        trips_cols = ['trip_id', 'route_id', 'shape_id']
        
        # Load stop times and trips
        bus_stop_times = pd.read_csv(os.path.join(bus_dir, 'stop_times.txt'))[stop_times_cols]
        train_stop_times = pd.read_csv(os.path.join(train_dir, 'stop_times.txt'))[stop_times_cols]
        stop_times = pd.concat([bus_stop_times, train_stop_times])
        bus_trips = pd.read_csv(os.path.join(bus_dir, 'trips.txt'))[trips_cols]
        train_trips = pd.read_csv(os.path.join(train_dir, 'trips.txt'))[trips_cols]
        trips = pd.concat([bus_trips, train_trips])
        
        # Load shapes for visualization
        bus_shapes = pd.read_csv(os.path.join(bus_dir, 'shapes.txt'))
        train_shapes = pd.read_csv(os.path.join(train_dir, 'shapes.txt'))
        self.shapes = pd.concat([bus_shapes, train_shapes])
        
        # Build network connections
        connections = stop_times.merge(trips[['trip_id', 'route_id']], on='trip_id')
        self._build_connection_maps(connections)
        
        # Store trip shape mappings for visualization
        self.trip_shapes = dict(zip(trips['route_id'], trips['shape_id']))

    def _build_connection_maps(self, connections):
        self.stop_routes = defaultdict(set)
        self.route_stops = defaultdict(set)
        self.connected_stops = defaultdict(set)
        
        # Convert connections to numpy arrays
        connections_array = connections[['stop_id', 'route_id']].values
        
        # Process all connections
        for stop_id, route_id in connections_array:
            stop_id_str = str(stop_id)
            route_id_str = str(route_id)
            self.stop_routes[stop_id_str].add(route_id_str)
            self.route_stops[route_id_str].add(stop_id_str)
        
        # Build connected stops map
        for route_id, stops in self.route_stops.items():
            stops_list = list(stops)
            for i, stop_id in enumerate(stops_list):
                # Add all other stops on this route as connections
                self.connected_stops[stop_id].update(
                    stops_list[:i] + stops_list[i+1:]
                )
        
        # Build spatial index using cKDTree
        self.kdtree = cKDTree(self.stop_coords)
        
        self.common_routes = {}
        for stop_id in self.stop_routes:
            connected_stops = self.connected_stops[stop_id]
            for other_stop in connected_stops:
                pair_key = tuple(sorted([stop_id, other_stop]))
                if pair_key not in self.common_routes:
                    self.common_routes[pair_key] = (
                        self.stop_routes[stop_id] & 
                        self.stop_routes[other_stop]
                    )

    def _build_shape_cache(self):
        """Build cache of shape points for route visualization"""
        self.logger.info("Building shape cache")
        shape_start = time.time()
        
        # Sort shapes and cache
        sorted_shapes = self.shapes.sort_values(['shape_id', 'shape_pt_sequence'])
        grouped_shapes = sorted_shapes.groupby('shape_id')
        self.route_shapes = {}
        for route_id, shape_id in self.trip_shapes.items():
            if shape_id in grouped_shapes.groups:
                shape_points = (grouped_shapes.get_group(shape_id)
                            [['shape_pt_lat', 'shape_pt_lon']]
                            .values.tolist())
                self.route_shapes[str(route_id)] = shape_points
        
        self.logger.info(f"Shape cache built in {time.time() - shape_start:.2f} seconds")

    def find_nearest_stops(self, lat, lon, radius=500, limit=3):
        # Convert radius to degrees
        radius_deg = radius / 111320
        
        nearby_indices = self.kdtree.query_ball_point([lat, lon], radius_deg)
        
        nearby_stops = []
        for idx in nearby_indices:
            stop = self.stop_info[idx]
            dist = geodesic((lat, lon), (stop["stop_lat"], stop["stop_lon"])).meters
            if dist <= radius:
                nearby_stops.append((stop, dist))
        
        return sorted(nearby_stops, key=lambda x: x[1])[:limit]

    def _find_multimodal_route(self, start_stops, end_stops):
        """Optimized A* search implementation with improved heuristics and pruning"""
        MAX_DIST = 30000  # 30km limit
        MAX_NODES = 5000
        
        # Compute end coordinates as numpy array
        end_coords = np.array([[float(stop['stop_lat']), float(stop['stop_lon'])] 
                            for stop, _ in end_stops])
        end_stop_ids = {str(stop['stop_id']) for stop, _ in end_stops}
        
        # Calculate center point of end stops for directional heuristic
        end_center = np.mean(end_coords, axis=0)
        
        # Find closest end stop for heuristic estimates
        closest_end_coord = end_coords[np.argmin([
            np.sum((coord - end_center) ** 2) for coord in end_coords
        ])]
        
        def calculate_heuristic(coord):
            return geodesic(coord, closest_end_coord).meters * 0.9 
        
        # Initialize priority queue with start stops
        queue = []
        visited = set()
        counter = 0
        g_scores = {}  # Track best known g_scores
        
        for start_stop, start_dist in start_stops:
            start_id = str(start_stop['stop_id'])
            start_coord = np.array([float(start_stop['stop_lat']), float(start_stop['stop_lon'])])
            h_score = calculate_heuristic(start_coord)
            g_scores[start_id] = start_dist
            
            entry = (
                start_dist + h_score, 
                counter,
                start_dist,
                start_id,
                [start_id],
                []
            )
            heapq.heappush(queue, entry)
            counter += 1
        
        nodes_explored = 0
        last_progress = 0
        best_end_distance = float('inf')
        
        while queue and nodes_explored < MAX_NODES:
            nodes_explored += 1
            
            f_score, _, g_score, current_stop_id, path, transfers = heapq.heappop(queue)
            
            # Early termination check
            if g_score > best_end_distance:
                continue
            
            if current_stop_id in visited or g_score > MAX_DIST:
                continue
                
            visited.add(current_stop_id)
            
            # Update progress logging
            if nodes_explored - last_progress >= 1000:
                self.logger.info(f"Explored {nodes_explored} nodes...")
                last_progress = nodes_explored
            
            if current_stop_id in end_stop_ids:
                if g_score < best_end_distance:
                    best_end_distance = g_score
                return path, transfers
            
            current_stop = self.stop_info_lookup[current_stop_id]
            current_coord = np.array([float(current_stop['stop_lat']), float(current_stop['stop_lon'])])
            
            # Skip if current path is already worse than best known
            if g_score >= best_end_distance:
                continue
            
            # Check if heading in the wrong direction
            if len(path) > 2:
                direction_to_end = end_center - current_coord
                current_direction = current_coord - np.array([float(self.stop_info_lookup[path[-2]]['stop_lat']),
                                                            float(self.stop_info_lookup[path[-2]]['stop_lon'])])
                if np.dot(direction_to_end, current_direction) < 0:
                    continue
            
            # Process transit connections with optimisations
            for next_stop_id in self.connected_stops[current_stop_id]:
                if next_stop_id in visited:
                    continue
                
                next_stop = self.stop_info_lookup[next_stop_id]
                next_coord = np.array([float(next_stop['stop_lat']), float(next_stop['stop_lon'])])
                
                # Quick distance check before detailed calculations
                approx_dist = np.sum((next_coord - current_coord) ** 2) ** 0.5 * 111320
                if approx_dist > 5000:  # Skip if distance too far
                    continue
                
                pair_key = tuple(sorted([current_stop_id, next_stop_id]))
                common_routes = self.common_routes.get(pair_key, set())
                
                for route_id in common_routes:
                    route_info = self.route_info[route_id]
                    dist = geodesic(current_coord, next_coord).meters
                    new_g_score = g_score + dist
                    
                    # Skip if already have a better path to this stop
                    if next_stop_id in g_scores and new_g_score >= g_scores[next_stop_id]:
                        continue
                    
                    if new_g_score <= MAX_DIST:
                        g_scores[next_stop_id] = new_g_score
                        new_transfers = transfers + [{
                            'type': 'transit',
                            'mode': route_info['mode'],
                            'route': route_info['route_short_name'],
                            'route_id': route_id,
                            'from_stop': current_stop_id,
                            'to_stop': next_stop_id,
                            'from_lat': float(current_stop['stop_lat']),
                            'from_lon': float(current_stop['stop_lon']),
                            'to_lat': float(next_stop['stop_lat']),
                            'to_lon': float(next_stop['stop_lon'])
                        }]
                        
                        h_score = calculate_heuristic(next_coord)
                        heapq.heappush(queue, (
                            new_g_score + h_score,
                            counter,
                            new_g_score,
                            next_stop_id,
                            path + [next_stop_id],
                            new_transfers
                        ))
                        counter += 1
            
            # Process walking transfers more efficiently
            if len(transfers) == 0 or transfers[-1]['type'] != 'walking':
                nearby_idx = self.kdtree.query_ball_point(
                    current_coord, 
                    r=500 / 111320
                )
                
                for idx in nearby_idx:
                    next_stop_id = self.stop_ids[idx]
                    if next_stop_id in visited:
                        continue
                    
                    next_stop = self.stop_info_lookup[next_stop_id]
                    if next_stop['mode'] != current_stop['mode']:
                        next_coord = self.stop_coords[idx]
                        walk_dist = geodesic(current_coord, next_coord).meters
                        new_g_score = g_score + walk_dist
                        
                        if next_stop_id in g_scores and new_g_score >= g_scores[next_stop_id]:
                            continue
                        
                        if new_g_score <= MAX_DIST:
                            g_scores[next_stop_id] = new_g_score
                            new_transfers = transfers + [{
                                'type': 'walking',
                                'from_lat': float(current_stop['stop_lat']),
                                'from_lon': float(current_stop['stop_lon']),
                                'to_lat': float(next_stop['stop_lat']),
                                'to_lon': float(next_stop['stop_lon']),
                                'distance': walk_dist
                            }]
                            
                            h_score = calculate_heuristic(next_coord)
                            heapq.heappush(queue, (
                                new_g_score + h_score,
                                counter,
                                new_g_score,
                                next_stop_id,
                                path + [next_stop_id],
                                new_transfers
                            ))
                            counter += 1
        
        return None, None

    def _get_route_segment(self, route_id, start_stop_id, end_stop_id):
        """Get shape points for route visualization"""
        if str(route_id) not in self.route_shapes:
            return None
            
        # Get stop coords
        start_stop = self.stop_info_lookup[str(start_stop_id)]
        end_stop = self.stop_info_lookup[str(end_stop_id)]
        start_coord = [float(start_stop['stop_lat']), float(start_stop['stop_lon'])]
        end_coord = [float(end_stop['stop_lat']), float(end_stop['stop_lon'])]
        
        # Get shape points and find closest points
        shape_points = self.route_shapes[str(route_id)]
        start_idx = min(range(len(shape_points)), 
                       key=lambda i: ((shape_points[i][0] - start_coord[0])**2 + 
                                    (shape_points[i][1] - start_coord[1])**2))
        end_idx = min(range(len(shape_points)), 
                     key=lambda i: ((shape_points[i][0] - end_coord[0])**2 + 
                                  (shape_points[i][1] - end_coord[1])**2))
        
        # Return ordered points
        if start_idx <= end_idx:
            return shape_points[start_idx:end_idx + 1]
        else:
            return shape_points[end_idx:start_idx + 1][::-1]

    def get_walking_route(self, from_lat, from_lon, to_lat, to_lon):
        client = openrouteservice.Client(key=self.api_key)
        coords = [[from_lon, from_lat], [to_lon, to_lat]]
        try:
            return client.directions(coordinates=coords, profile='foot-walking', format='geojson')
        except Exception as e:
            self.logger.error(f"Error getting walking route: {e}")
            return None

    def visualize_route(self, start_coords, end_coords, path, transfers):
        viz_start = time.time()
        self.logger.info("Starting visualization...")
        
        # Initialise map
        m = folium.Map(location=[start_coords[0], start_coords[1]], zoom_start=13)
        
        # Color schemes for different modes
        colors = {
            '4': {  # bus colors
                'blue': '#0066CC', 'red': '#CC0000', 'green': '#009933',
                'purple': '#660099', 'orange': '#FF6600',
            },
            '2': {  # train colors
                'blue': '#000066', 'red': '#660000', 'green': '#006600',
                'purple': '#330066', 'orange': '#CC3300',
            }
        }
        
# Add start and end markers
        folium.Marker(
            [start_coords[0], start_coords[1]], 
            popup='Start', 
            icon=folium.Icon(color='green')
        ).add_to(m)
        
        folium.Marker(
            [end_coords[0], end_coords[1]], 
            popup='End', 
            icon=folium.Icon(color='red')
        ).add_to(m)
        
        used_colors = {'2': 0, '4': 0}
        
        # Draw route segments
        for transfer in transfers:
            if transfer['type'] == 'walking':
                # Get walking route from OpenRouteService
                walk_route = self.get_walking_route(
                    transfer['from_lat'], transfer['from_lon'],
                    transfer['to_lat'], transfer['to_lon']
                )
                
                if walk_route:
                    folium.GeoJson(
                        walk_route,
                        style_function=lambda x: {
                            'color': '#00FF00', 
                            'weight': 3, 
                            'opacity': 0.7
                        },
                        popup=f'Walking ({transfer.get("distance", 0):.0f}m)'
                    ).add_to(m)
                else:
                    # Fallback to straight line if API fails
                    folium.PolyLine(
                        locations=[[transfer['from_lat'], transfer['from_lon']],
                                [transfer['to_lat'], transfer['to_lon']]],
                        weight=3,
                        color='#00FF00',
                        popup=f'Walking ({transfer.get("distance", 0):.0f}m)',
                        opacity=0.7
                    ).add_to(m)
                    
            elif transfer['type'] == 'transit':
                segment_points = self._get_route_segment(
                    transfer['route_id'], 
                    transfer['from_stop'], 
                    transfer['to_stop']
                )
                
                if segment_points:
                    # Get mode name and color
                    mode = transfer['mode']
                    mode_names = {'2': 'Train', '4': 'Bus'}
                    mode_name = mode_names.get(mode, 'Transit')
                    
                    color_idx = used_colors[mode]
                    color = list(colors[mode].values())[color_idx % len(colors[mode])]
                    used_colors[mode] += 1
                    
                    # Draw route line
                    folium.PolyLine(
                        locations=segment_points, weight=4, color=color, popup=f"{mode_name} {transfer['route']}", opacity=0.8
                    ).add_to(m)
                    
                    # Add route label
                    if len(segment_points) >= 2:
                        mid_idx = len(segment_points) // 2
                        mid_point = segment_points[mid_idx]
                        folium.DivIcon(
                            html=f'<div style="background-color: {color}; color: white; padding: 3px 6px; border-radius: 3px; font-weight: bold;">{mode_name} {transfer["route"]}</div>',
                            icon_size=(70, 20), icon_anchor=(35, 10)
                        ).add_to(folium.Marker(mid_point).add_to(m))
        
        # Add transit stop markers
        for i, stop_id in enumerate(path):
            stop = self.stop_info_lookup[stop_id]
            
            # Determine marker type
            if i == 0:
                icon_color = 'green'
                prefix = 'Start'
            elif i == len(path) - 1:
                icon_color = 'red'
                prefix = 'End'
            else:
                icon_color = 'blue'
                prefix = 'Transfer'
            
            folium.Marker(
                [float(stop['stop_lat']), float(stop['stop_lon'])],
                popup=f"{prefix}: {stop['stop_name']}",
                icon=folium.Icon(color=icon_color, icon='info-sign')
            ).add_to(m)
        
        # Save map
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
        output_dir = os.path.join('..', 'mpt_data', 'maps')
        os.makedirs(output_dir, exist_ok=True)
        map_file = os.path.join(output_dir, f'multimodal_route_{timestamp}.html')
        m.save(map_file)
        
        return map_file

    def find_route(self, start_location, end_location):
        """Main routing function to find path between two locations"""
        self.logger.info(f"\nFinding route from {start_location} to {end_location}")
        
        # Geocode locations
        geolocator = Nominatim(user_agent="multimodal_router", timeout=10)
        
        # Add Melbourne context
        start_loc = geolocator.geocode(f"{start_location}, Melbourne, Victoria", exactly_one=True)
        end_loc = geolocator.geocode(f"{end_location}, Melbourne, Victoria", exactly_one=True)
        
        if not start_loc or not end_loc:
            return "Could not geocode one or both locations in Melbourne", None
        
        # Find nearest transit stops
        start_stops = self.find_nearest_stops(start_loc.latitude, start_loc.longitude, radius=1000, limit=5)
        end_stops = self.find_nearest_stops(end_loc.latitude, end_loc.longitude, radius=1000, limit=5)
        
        if not start_stops:
            return f"No transit stops found within 1km of {start_location}", None
        if not end_stops:
            return f"No transit stops found within 1km of {end_location}", None
        
        # Find optimal route
        path, transfers = self._find_multimodal_route(start_stops, end_stops)
        
        if not path:
            return "No route found", None
        
        # Add initial and final walking segments
        first_stop = self.stop_info_lookup[path[0]]
        last_stop = self.stop_info_lookup[path[-1]]
        
        initial_walk = {
            'type': 'walking',
            'from_lat': start_loc.latitude,
            'from_lon': start_loc.longitude,
            'to_lat': float(first_stop['stop_lat']),
            'to_lon': float(first_stop['stop_lon']),
            'distance': geodesic(
                (start_loc.latitude, start_loc.longitude),
                (first_stop['stop_lat'], first_stop['stop_lon'])
            ).meters,
            'description': f'Walk to {first_stop["stop_name"]}'
        }
        
        final_walk = {
            'type': 'walking',
            'from_lat': float(last_stop['stop_lat']),
            'from_lon': float(last_stop['stop_lon']),
            'to_lat': end_loc.latitude,
            'to_lon': end_loc.longitude,
            'distance': geodesic(
                (last_stop['stop_lat'], last_stop['stop_lon']),
                (end_loc.latitude, end_loc.longitude)
            ).meters,
            'description': f'Walk from {last_stop["stop_name"]} to destination'
        }
        
        transfers = [initial_walk] + transfers + [final_walk]
        
        # Generate directions text
        output_parts = []
        output_parts.append("Route found:") 
        total_walking = 0
        total_transfers = 0
        
        #Add directions
        for transfer in transfers:
            if transfer['type'] == 'walking':
                output_parts.append(f"{transfer.get('description', 'Walk')} ({transfer['distance']:.0f}m)")
                total_walking += transfer['distance']
            else:
                mode_names = {'2': 'Train', '4': 'Bus'}
                mode = mode_names.get(transfer['mode'], 'Transit')
                output_parts.append(f"Take {mode} {transfer['route']} to {self.stop_info_lookup[transfer['to_stop']]['stop_name']}")
                total_transfers += 1
        
        summary_parts = [
            f"Total transfers: {total_transfers}", f"Total walking distance: {total_walking:.0f}m" ]
        
        # Generate visualization
        map_file = self.visualize_route(
            (start_loc.latitude, start_loc.longitude), (end_loc.latitude, end_loc.longitude), path, transfers
        )
        
        # Combine all parts with HTML formatting
        formatted_output = "<br><br>".join([
            output_parts[0], 
            "<br>".join(output_parts[1:]),  
            "<br>".join(summary_parts)  
        ])
        
        return formatted_output, map_file
