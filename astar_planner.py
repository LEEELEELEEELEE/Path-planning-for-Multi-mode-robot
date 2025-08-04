# astar_planner.py
#Author: Shuyi Li
#University: The University of Manchester
#Project: AI_empowered pathplanning for multi-modes robots 
#Date: 04/08/2025
###############################################################################

from astar import AStar
import math

def is_safe(grid, x, y, z, radius=1):
    """Check surrounding cube of radius for obstacles."""
    X, Y, Z = len(grid), len(grid[0]), len(grid[0][0])
    for dx in range(-radius-1, radius+1):
        for dy in range(-radius-1, radius+1):
            for dz in range(-radius-1, radius+1):
                nx, ny, nz = x+dx, y+dy, z+dz
                if 0 <= nx < X and 0 <= ny < Y and 0 <= nz < Z:
                    if grid[nx][ny][nz] == 1:
                        return False
    return True

class PyBulletAStar3D(AStar):
    def __init__(self, grid, ai_score=None, robot_radius=1):
        self.grid     = grid
        self.ai_score = ai_score
        self.robot_radius = robot_radius          # pre-set safety/feasibility costs
        self.size_x   = len(grid)
        self.size_y   = len(grid[0])
        self.size_z   = len(grid[0][0])
        self.edge_mode = {}               # record drive/fly per edge
        self.ground_z = 0
    def is_valid(self, x, y, z):
        #Bounds + collision + safety.
        return (
            0 <= x < self.size_x and
            0 <= y < self.size_y and
            0 <= z < self.size_z and
            self.grid[x][y][z] == 0 and
            is_safe(self.grid, x, y, z, radius=self.robot_radius)
        )

    def heuristic_cost_estimate(self, a, b):
        #3D Euclidean distance.
        return math.sqrt(sum((u - v)**2 for u, v in zip(a, b)))

    def _neighbors(self, current_node, search_nodes):
        #Generate (pos,mode) → fold duplicates preferring drive.
        x,y,z = current_node.data
        raw = []

        # DRIVE neighbors (only on ground plane)
        # so z == ground_z
        if z == self.ground_z:
            for dx in (-1,0,1):
                for dy in (-1,0,1):
                    if dx==0 and dy==0: continue
                    nx,ny,nz = x+dx, y+dy, z
                    if self.is_valid(nx,ny,nz):
                        raw.append(((nx,ny,nz),"drive"))

        # Fly: full 3D mdoe driving
        for dx in (-1,0,1):
            for dy in (-1,0,1):
                for dz in (-1,0,1):
                    if dx==0 and dy==0 and dz==0: continue
                    nx,ny,nz = x+dx, y+dy, z+dz
                    if self.is_valid(nx,ny,nz):
                        raw.append(((nx,ny,nz),"fly"))

        # Deduplicate, keep drive if both modes exist
        unique = {}
        for pos,mode in raw:
            if pos not in unique or (unique[pos]=="fly" and mode=="drive"):
                unique[pos] = mode

        # Build SearchNodes & record mode
        result = []
        for pos,mode in unique.items():
            self.edge_mode[(current_node.data, pos)] = mode
            result.append(search_nodes[pos])
        return result

    def distance_between(self, a, b):
        """Combines safety (AI) + realistic energy/time costs for the recorded mode."""
        mode = self.edge_mode.get((a, b), "drive")

        # Euclidean base distance in m
        base = math.sqrt(sum((u - v)**2 for u, v in zip(a, b))) * 0.25  # account for grid resolution

        # Sample AI safety score along segment
        steps = 5
        total_s = 0
        for i in range(steps + 1):
            t = i / steps
            xi = int(a[0] + (b[0] - a[0]) * t)
            yi = int(a[1] + (b[1] - a[1]) * t)
            zi = int(a[2] + (b[2] - a[2]) * t)
            total_s += self.ai_score[xi][yi][zi]
        avg_s = total_s / (steps + 1)

        # Mode-specific realistic energy and time cost
        if mode == "drive": #1:18 scale RCE cars
            energy_cost = 2.284 * base
            time_cost = base / 8.33
        else:  # fly: DJI drone
            energy_cost = 6.646 * base
            time_cost = base / 18.0555

        # Final cost function 
        cost =  avg_s + 2 * energy_cost + time_cost
        return base + cost