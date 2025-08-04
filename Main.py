#Main.py (for the simulation setup and deploy AI_empowered A* Algorithm
#Author: Shuyi Li
#University: The University of Manchester
#Project: AI_empowered path planning for multi-modes robots 
#Date: 04/08/2025
###############################################################################
import pybullet as p
import pybullet_data
import time
import math
import heapq
import random
from astar_planner import PyBulletAStar3D

# Pybullet setup
resolution = 0.25  # Resolution of the grid in meters
def grid_to_world(i, j, k, resolution=resolution):
    return [i * resolution, j * resolution, k * resolution]

def create_obstacles(grid, resolution=resolution):
    obstacle_ids = []
    for i in range(len(grid)):
        for j in range(len(grid[0])):
            for k in range(len(grid[0][0])):
                if grid[i][j][k] == 1:
                    hx = random.uniform(0.3*resolution, 1.0*resolution)
                    hy = random.uniform(0.3*resolution, 1.0*resolution)
                    hz = random.uniform(0.3*resolution, 1.0*resolution)
                    colBoxId = p.createCollisionShape(p.GEOM_BOX, halfExtents=[hx, hy, hz])
                    x, y, z = grid_to_world(i, j, k, resolution)
                    z += hz
                    body_id = p.createMultiBody(0, colBoxId, basePosition=[x, y, z])
                    obstacle_ids.append(body_id)
    return obstacle_ids

def visualize_ai_score_regions(ai_score, resolution=1.0):
    for i in range(len(ai_score)):
        for j in range(len(ai_score[0])):
            for k in range(len(ai_score[0][0])):
                score = ai_score[i][j][k]
                if score < 2:
                    continue  # Skip low-cost zones

                # Map score to color and transparency
                if score >= 10:
                    color = [0, 0, 1, 0.3]  # Blue: WINDY
                else:
                    color = [1, 0.5, 0, 0.3]  # Orange: SLOPE

                # Create semi-transparent cube
                visual_shape_id = p.createVisualShape(
                    shapeType=p.GEOM_BOX,
                    halfExtents=[0.5 * resolution] * 3,
                    rgbaColor=color
                )

                pos = grid_to_world(i, j, k, resolution)
                p.createMultiBody(
                    baseMass=0,
                    baseVisualShapeIndex=visual_shape_id,
                    basePosition=pos,
                    useMaximalCoordinates=True
                )

def main():
    
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    p.loadURDF("plane.urdf")
    p.resetDebugVisualizerCamera(
        cameraDistance=5,
        cameraYaw=5,
        cameraPitch=-5,
        cameraTargetPosition=[5, 5, 5]
    )

    # Define 3D grid with obstacles
    size_x, size_y, size_z = int(10 / resolution), int(10 / resolution), int(10 / resolution)
    grid = [[[0 for _ in range(size_z)] for _ in range(size_y)] for _ in range(size_x)]
    for x, y, z in random.sample([(i, j, k) for i in range(size_x) for j in range(size_y) for k in range(size_z)], 100):
        grid[x][y][z] = 1  # vertical obstacle
    # INSERT ai_score DEFINITION HERE
    ai_score = [[[1 for _ in range(size_z)] for _ in range(size_y)] for _ in range(size_x)]

    # Example: Terrain simulation for AI scoring
    for i in range(size_x):
        for j in range(size_y):
            for k in range(size_z):
                if 12 <= i <= 25 and 20 <= j <= 25 and 0 <= k <= 7:
                    ai_score[i][j][k] = 10  # windy zone
                if 0 <= i <= 16 and 10 <= j <= 20 and 0 <= k <= 1:
                    ai_score[i][j][k] += 2  # sloped region

    obstacle_ids = create_obstacles(grid)
    print("Obstacle summary from grid:")
    for z in range(size_z):
        for y in range(size_y):
            for x in range(size_x):
                if grid[x][y][z] == 1:
                    print(f"Obstacle at: ({x}, {y}, {z})")

    start = (0, 0, 0)
    goal = (19, 27, 10)
    solver = PyBulletAStar3D(grid, ai_score=ai_score, robot_radius=1)
    
    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)

    visualize_ai_score_regions(ai_score, resolution)
    
    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)

    path = list(solver.astar(start, goal))
    
    # Extract per-edge modes & find morph point(s)
    mode_list = [
      solver.edge_mode.get((path[i],path[i+1]), "drive")
      for i in range(len(path)-1)
    ]
    
    # debugging
    switch_penalty = 3.0
    total_cost = 0.0
    print("Edge costs (with α·S + β·E + γ·T):")
    for i in range(len(path)-1):
        a, b = path[i], path[i+1]
        c = solver.distance_between(a, b)
        print(f"  {a}->{b} [{mode_list[i]}] : travel_cost = {c:.3f}")
        total_cost += c

    # now add penalty for each mode switch
    num_switches = 0
    for i in range(1, len(mode_list)):
        if mode_list[i] != mode_list[i-1]:
            print(f"  Switch {mode_list[i-1]}→{mode_list[i]}: +{switch_penalty:.1f}s")
            total_cost += switch_penalty
            num_switches += 1

    print(f"Total segments: {len(path)-1}, switches: {num_switches}")
    print(f"total path cost include switch: {total_cost:.3f}")

    # Find where mode switches
    morphs = [
      (path[i], mode_list[i-1], mode_list[i])
      for i in range(1,len(mode_list))
      if mode_list[i] != mode_list[i-1]
    ]
    print("Mode morphs:", morphs)

    # Draw visible path line
    for i in range(len(path)-1):
        color = [1,0,0] if mode_list[i]=="drive" else [0,1,0]
        p.addUserDebugLine(
           grid_to_world(*path[i]), grid_to_world(*path[i+1]),
           lineColorRGB=color, lineWidth=2, lifeTime=0
        )

    if path is None:
        print("No path found")
        return

    print("3D Path:", path)

    # Create robot
    colBoxId = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.1, 0.1, 0.1])
    robot_id = p.createMultiBody(baseMass=1, baseCollisionShapeIndex=colBoxId, basePosition=grid_to_world(*start))
    # Speed define
    mode_speeds = {
    "drive": 8.33,      # m/s
    "fly": 18.0555      # m/s
}

    for waypoint in path:
        if grid[waypoint[0]][waypoint[1]][waypoint[2]] == 1:
            print(f"ERROR: Path includes an obstacle cell: {waypoint}")
    

    # Move through path
    prev_mode = mode_list[0]
    for i in range(len(path) - 1):
        curr_mode = mode_list[i]

        # If mode switched, wait for morphing
        if curr_mode != prev_mode:
            print(f" Switching {prev_mode}→{curr_mode}, waiting {switch_penalty}s")
            morph_steps = int(switch_penalty * 60)
            for _ in range(morph_steps):
                p.stepSimulation()
                time.sleep(1.0 / 60.0)
        prev_mode = curr_mode

        start_pos = grid_to_world(*path[i])
        end_pos = grid_to_world(*path[i + 1])

        dist = math.sqrt(sum((a - b) ** 2 for a, b in zip(start_pos, end_pos)))
        mode_speed = mode_speeds[curr_mode]
        duration = dist / mode_speed
        steps = max(1, int(duration * 60))

        for s in range(steps):
            interp_pos = [
                start_pos[0] + (end_pos[0] - start_pos[0]) * s / steps,
                start_pos[1] + (end_pos[1] - start_pos[1]) * s / steps,
                start_pos[2] + (end_pos[2] - start_pos[2]) * s / steps + 0.1
            ]
            p.resetBasePositionAndOrientation(robot_id, interp_pos, [0, 0, 0, 1])
            p.stepSimulation()
            time.sleep(1. / 60.)

    time.sleep(3)
    p.disconnect()

if __name__ == "__main__":
    main()
