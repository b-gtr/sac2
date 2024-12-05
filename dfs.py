import os
import math
import threading
import carla
import numpy as np
import torch
import random
import time
from abc import ABC, abstractmethod
from soft_actor_critic import ParamsPool
from replay_buffer import ReplayBuffer, Transition

# Trainingsparameter
NUM_EPISODES = 1000
MAX_STEPS_PER_EPISODE = 1000
BATCH_SIZE = 64
LOAD_MODEL = False  # Auf True setzen, wenn ein gespeichertes Modell geladen werden soll

# Gerät initialisieren
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_gpus = torch.cuda.device_count()
print(f"Number of GPUs available: {num_gpus}")


class Sensor(ABC):
    def __init__(self, vehicle):
        self.vehicle = vehicle
        self.sensor = None
        self.history = []

    @abstractmethod
    def listen(self):
        pass

    def clear_history(self):
        self.history.clear()

    def destroy(self):
        if self.sensor is not None:
            try:
                self.sensor.destroy()
            except RuntimeError as e:
                print(f"Fehler beim Zerstören des Sensors: {e}")
        
    def get_history(self):
        return self.history

class CollisionSensor(Sensor):
    def __init__(self, vehicle, blueprint_library, world):
        super().__init__(vehicle)
        collision_bp = blueprint_library.find('sensor.other.collision')
        self.sensor = world.spawn_actor(collision_bp, carla.Transform(), attach_to=self.vehicle)
        
    def _on_collision(self, event):
        self.history.append(event)

    def listen(self):
        self.sensor.listen(self._on_collision)

class LaneInvasionSensor(Sensor):
    def __init__(self, vehicle, blueprint_library, world):
        super().__init__(vehicle)
        lane_invasion_bp = blueprint_library.find('sensor.other.lane_invasion')
        self.sensor = world.spawn_actor(lane_invasion_bp, carla.Transform(), attach_to=self.vehicle)

    def _on_lane_invasion(self, event):
        self.history.append(event)

    def listen(self):
        self.sensor.listen(self._on_lane_invasion)        

class GnssSensor(Sensor):
    def __init__(self, vehicle, blueprint_library, world):
        super().__init__(vehicle)
        gnss_bp = blueprint_library.find('sensor.other.gnss')
        self.sensor = world.spawn_actor(gnss_bp, carla.Transform(), attach_to=self.vehicle)
        self.current_gnss = None

    def _on_gnss_event(self, event):
        self.current_gnss = event

    def listen(self):
        self.sensor.listen(self._on_gnss_event)
    
    def get_current_gnss(self):
        return self.current_gnss

class CameraSensor(Sensor):
    def __init__(self, vehicle, blueprint_library, world, image_processor_callback):
        super().__init__(vehicle)
        camera_bp = blueprint_library.find('sensor.camera.semantic_segmentation')
        camera_bp.set_attribute('image_size_x', '640')
        camera_bp.set_attribute('image_size_y', '480')
        camera_bp.set_attribute('fov', '110')
        camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
        self.sensor = world.spawn_actor(camera_bp, camera_transform, attach_to=self.vehicle)
        self.image_processor_callback = image_processor_callback

    def listen(self):
        self.sensor.listen(self.image_processor_callback)

class CarlaEnv:
    def __init__(self):
        self.client = carla.Client('localhost', 2000)
        self.client.set_timeout(10.0)
        self.client.load_world('Town01')
        self.world = self.client.get_world()
        self.blueprint_library = self.world.get_blueprint_library()
        self.spawn_points = self.world.get_map().get_spawn_points()

        self.vehicle = None
        self.camera_sensor = None
        self.collision_sensor = None
        self.lane_invasion_sensor = None
        self.gnss_sensor = None

        self.image_lock = threading.Lock()
        self.running = True

        # Synchroner Modus
        self.original_settings = self.world.get_settings()
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05  # 20 FPS
        self.world.apply_settings(settings)

        self.latest_image = None
        self.agent_image = None

        self.reset_environment()

    def reset_environment(self):
        self._clear_sensors()

        # Fahrzeug zerstören, falls es existiert
        if self.vehicle is not None:
            self.vehicle.destroy()
            self.vehicle = None

        self.collision_sensor = None
        self.lane_invasion_sensor = None
        self.gnss_sensor = None
        self.camera_sensor = None

        # Fahrzeug an zufälligem Startpunkt spawnen
        vehicle_bp = self.blueprint_library.filter('vehicle.lincoln.mkz_2017')[0]
        self.spawn_point = self.spawn_points[0]
        self.spawn_rotation = self.spawn_point.rotation
        self.vehicle = self.world.spawn_actor(vehicle_bp, self.spawn_point)

        # Sensoren anbringen
        self.setup_sensors()

        # Warten, bis die Sensoren initialisiert sind
        for _ in range(10):
            self.world.tick()

    def setup_sensors(self):
        # Kamera-Sensor
        self.camera_sensor = CameraSensor(self.vehicle, self.blueprint_library, self.world, self.process_image)
        self.camera_sensor.listen()

        # Kollisionssensor
        self.collision_sensor = CollisionSensor(self.vehicle, self.blueprint_library, self.world)
        self.collision_sensor.listen()

        # Spurverletzungssensor
        self.lane_invasion_sensor = LaneInvasionSensor(self.vehicle, self.blueprint_library, self.world)
        self.lane_invasion_sensor.listen()

        # GNSS-Sensor
        self.gnss_sensor = GnssSensor(self.vehicle, self.blueprint_library, self.world)
        self.gnss_sensor.listen()

    def _clear_sensors(self):
        if self.camera_sensor is not None:
            self.camera_sensor.destroy()
        if self.collision_sensor is not None:
            self.collision_sensor.destroy()
        if self.lane_invasion_sensor is not None:
            self.lane_invasion_sensor.destroy()
        if self.gnss_sensor is not None:
            self.gnss_sensor.destroy()
        self.latest_image = None
        self.agent_image = None

    def process_image(self, image):
        # Labels für den Agenten erhalten
        image.convert(carla.ColorConverter.Raw)
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = array.reshape((image.height, image.width, 4))
        labels = array[:, :, 2]  # Labels aus dem roten Kanal extrahieren

        # Labels normalisieren
        with self.image_lock:
            self.agent_image = labels / 22.0  # Normalisierung auf [0, 1]

    def get_vehicle_speed(self):
        vel = self.vehicle.get_velocity()
        return math.sqrt(vel.x**2 + vel.y**2 + vel.z**2)

    def destroy(self):
        # Ursprüngliche Einstellungen wiederherstellen
        self.world.apply_settings(self.original_settings)

        # Akteure bereinigen
        if self.camera_sensor is not None:
            self.camera_sensor.destroy()
        if self.collision_sensor is not None:
            self.collision_sensor.destroy()
        if self.lane_invasion_sensor is not None:
            self.lane_invasion_sensor.destroy()
        if self.gnss_sensor is not None:
            self.gnss_sensor.destroy()
        if self.vehicle is not None:
            self.vehicle.destroy()

    # New Function Added: Get Lane Center and Lateral Offset
    def get_lane_center_and_offset(self):
        """
        Retrieves the center of the current lane and calculates the lateral offset of the vehicle from the lane center.

        Returns:
            lane_center (carla.Location): The location of the lane center.
            lateral_offset (float): The lateral distance from the vehicle to the lane center.
                                     Positive if to the right, negative if to the left.
        """
        vehicle_transform = self.vehicle.get_transform()
        vehicle_location = vehicle_transform.location

        map = self.world.get_map()
        waypoint = map.get_waypoint(vehicle_location, project_to_road=True)
        if not waypoint:
            print("No waypoint found for the vehicle location.")
            return None, None

        # Lane center
        lane_center = waypoint.transform.location

        # Calculate lateral offset
        dx = vehicle_location.x - lane_center.x
        dy = vehicle_location.y - lane_center.y

        lane_heading = math.radians(waypoint.transform.rotation.yaw)
        lane_direction = carla.Vector3D(math.cos(lane_heading), math.sin(lane_heading), 0)
        perpendicular_direction = carla.Vector3D(-lane_direction.y, lane_direction.x, 0)

        lateral_offset = dx * perpendicular_direction.x + dy * perpendicular_direction.y

        return lane_center, lateral_offset

def train_agent(env, agent, replay_buffer, num_episodes=1000, max_steps_per_episode=1000, batch_size=64):
    try:
        for episode in range(num_episodes):
            if not env.running:
                break
            env.reset_environment()

            direction_vector = env.spawn_rotation.get_forward_vector()

            destination = env.spawn_point.location + direction_vector * 40

            previous_distance = None
            episode_reward = 0
            termination_reason = None

            for step in range(max_steps_per_episode):
                if not env.running:
                    break

                with env.image_lock:
                    current_agent_image = env.agent_image.copy() if env.agent_image is not None else None

                current_gnss = env.gnss_sensor.get_current_gnss()
                if current_agent_image is None or current_gnss is None:
                    env.world.tick()
                    continue

                transform = env.vehicle.get_transform()
                location = transform.location
                rotation = transform.rotation
                yaw = math.radians(rotation.yaw)

                # Replace Existing Lane Center and Lateral Deviation Calculation
                lane_center, lateral_offset = env.get_lane_center_and_offset()
                if lane_center is None:
                    env.world.tick()
                    continue

                distance_to_destination = location.distance(destination)

                if previous_distance is None:
                    previous_distance = distance_to_destination

                # Update scalars to include distance_to_destination and lateral_offset
                scalars = np.array([distance_to_destination, lateral_offset])

                action = agent.act(current_agent_image, scalars)

                steer = float(action[0])
                steer = steer / 2
                throttle = float(action[1])
                throttle = 0.5 * (1 + throttle)
                control = carla.VehicleControl(
                    steer=np.clip(steer, -1.0, 1.0),
                    throttle=np.clip(throttle, 0.0, 1.0)
                )
                env.vehicle.apply_control(control)

                env.world.tick()

                with env.image_lock:
                    next_agent_image = env.agent_image.copy() if env.agent_image is not None else None

                transform = env.vehicle.get_transform()
                location = transform.location
                rotation = transform.rotation
                yaw = math.radians(rotation.yaw)

                speed = env.get_vehicle_speed()

                deviation_threshold = 0.7
                deviation_penalty_scale = 4.0

                if len(env.collision_sensor.get_history()) > 0:
                    reward = -30
                    done = True
                    termination_reason = 'collision'
                elif step >= max_steps_per_episode - 1:
                    reward = -10
                    done = True
                    termination_reason = 'timeout'
                else:
                    if abs(lateral_offset) <= deviation_threshold:
                        r_lane_centering = 1.0 / (abs(lateral_offset) + 0.1)
                    else:
                        r_lane_centering = -deviation_penalty_scale * (abs(lateral_offset) - deviation_threshold)

                    v = speed * 3.6
                    v_target = 20
                    r_speed = 1 - min(1, abs(v - v_target) / 5)

                    map = env.world.get_map()
                    waypoint = map.get_waypoint(location)
                    next_waypoints = waypoint.next(2.0)
                    if next_waypoints:
                        next_waypoint = next_waypoints[0]
                    else:
                        next_waypoint = waypoint

                    wp_location = next_waypoint.transform.location
                    dx = wp_location.x - location.x
                    dy = wp_location.y - location.y
                    desired_yaw = math.atan2(dy, dx)
                    epsilon = desired_yaw - yaw
                    epsilon = (epsilon + math.pi) % (2 * math.pi) - math.pi

                    r_heading = - (abs(epsilon) / 3) ** 2

                    if distance_to_destination < previous_distance:
                        r_traveled = 1
                    else:
                        r_traveled = -0.1

                    r_overspeed = -5 if v > 25 else 0

                    reward = r_lane_centering + r_speed + r_heading + r_traveled + r_overspeed
                    done = False

                episode_reward += reward

                print(f"Episode {episode+1}, Schritt {step}, Belohnung: {reward:.2f}, Gesamtbelohnung: {episode_reward:.2f}")

                transition = Transition(
                    img=current_agent_image,
                    scalars=scalars,
                    a=action,
                    r=reward,
                    n_img=next_agent_image,
                    n_scalars=scalars,
                    d=done
                )
                replay_buffer.push(transition)

                if step % 25 == 0 and replay_buffer.ready_for(batch_size):
                    for _ in range(25):
                        batch = replay_buffer.sample(batch_size)
                        agent.update_networks(batch)

                if done:
                    break

                previous_distance = distance_to_destination

            print(f'Episode {episode+1}, Gesamtbelohnung: {episode_reward:.2f}')
            if (episode + 1) % 50 == 0:
                agent.save_model('model_params.pth')
                print(f'Modellparameter nach Episode {episode+1} gespeichert.')

        print('Training abgeschlossen.')

    finally:
        env.destroy()


def main():
    env = CarlaEnv()
    latent_dim = 95  # Same as LATENT_SPACE in vae.py
    scalar_dim = 2
    action_dim = 2

    agent = ParamsPool(
        latent_dim=latent_dim,
        scalar_dim=scalar_dim,
        action_dim=action_dim,
        device=device
    )
    replay_buffer = ReplayBuffer(capacity=15000, device=device)

    load_model = input("Möchten Sie gespeicherte Modellparameter laden? (j/n): ")
    if load_model.lower() == 'j':
        agent.load_model('model_params.pth')
        print('Modellparameter geladen.')

    train_agent(env, agent, replay_buffer,
                num_episodes=NUM_EPISODES,
                max_steps_per_episode=MAX_STEPS_PER_EPISODE,
                batch_size=BATCH_SIZE)

if __name__ == "__main__":
    main()
