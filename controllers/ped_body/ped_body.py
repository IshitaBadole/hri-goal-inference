# controllers/ped_body/ped_body.py
from controller import Supervisor
import math

BODY_Z   = 0.72
VIS_Z    = 1.27
STEP_M   = 0.04
RADIUS_M = 0.25
STOP_EPS = 0.05
RAY_EPS  = 0.01

class PedBody(Supervisor):
    def __init__(self):
        super().__init__()
        self.dt   = int(self.getBasicTimeStep())
        self.body = self.getSelf()
        self.tr   = self.body.getField("translation")
        self.rot  = self.body.getField("rotation")

        # optional visual
        self.vis     = self.getFromDef("PED_VIS")
        self.vis_tr  = self.vis.getField("translation") if self.vis else None
        self.vis_rot = self.vis.getField("rotation") if self.vis else None

        # Waypoints
        self.goals = [(-5.0, -3.0), (-3.5, -2.0), (-6.5, -2.6), (-5.2, -4.8)]
        self.gi = 0

        # ---- Raycast function: may be on Supervisor or on Node ----
        self._supervisor_ray = getattr(self, "rayCast", None)   # Supervisor.rayCast?
        
        self._root = self.getRoot()
        self._node_ray = getattr(self._root, "rayCast", None) or getattr(self.body, "rayCast", None)

    def _rc(self, p_from, p_to):
        """Unified raycast: returns dict with 'point','normal','node' or None."""
        hit = None
        if self._supervisor_ray:
            hit = self._supervisor_ray(p_from, p_to)
        elif self._node_ray:
            hit = self._node_ray(p_from, p_to)
        if hit and "point" in hit:
            return hit
        return None

    def _set_pose(self, x, y, yaw):
        self.tr.setSFVec3f([x, y, BODY_Z])
        self.rot.setSFRotation([0, 0, 1, yaw])
        if self.vis_tr:  self.vis_tr.setSFVec3f([x, y, VIS_Z])
        if self.vis_rot: self.vis_rot.setSFRotation([0, 0, 1, yaw])

    def _advance_toward(self, x, y, tx, ty):
        dx, dy = tx - x, ty - y
        dist = math.hypot(dx, dy)
        if dist < 1e-9:
            return x, y, 0.0, True
        nx, ny = dx/dist, dy/dist
        step = min(STEP_M, dist)
        nxp, nyp = x + nx*step, y + ny*step

        # If raycast exists, stop before walls
        if self._supervisor_ray or self._node_ray:
            hit = self._rc([x, y, BODY_Z], [nxp, nyp, BODY_Z])
            if hit:
                hx, hy, _ = hit["point"]
                back = RADIUS_M + RAY_EPS
                stop_x = hx - nx*back
                stop_y = hy - ny*back
                # avoid overshooting backwards
                if (stop_x - x)*nx + (stop_y - y)*ny < 0:
                    stop_x, stop_y = x, y
                yaw = math.atan2(ny, nx)
                return stop_x, stop_y, yaw, True

        yaw = math.atan2(ny, nx)
        arrived = (dist - step) <= STOP_EPS
        return nxp, nyp, yaw, arrived

    def run(self):
        x, y, _ = self.tr.getSFVec3f()
        yaw = 0.0
        self._set_pose(x, y, yaw)

        while self.step(self.dt) != -1:
            tx, ty = self.goals[self.gi]
            x, y, yaw, hit_or_arrived = self._advance_toward(x, y, tx, ty)
            self._set_pose(x, y, yaw)
            if hit_or_arrived and math.hypot(tx - x, ty - y) <= STOP_EPS + 0.05:
                self.gi = (self.gi + 1) % len(self.goals)

if __name__ == "__main__":
    PedBody().run()