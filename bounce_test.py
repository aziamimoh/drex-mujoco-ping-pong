"""
Measure how the ball bounces in pingpong.xml for several contact settings, to find
settings where bounces lose a little energy (restitution < 1) instead of gaining it.

    python bounce_test.py

Each row overrides every geom's solref (timeconst, dampratio) and optionally the timestep,
then runs three headless tests:
  drop   - ball dropped 30 cm onto the table; restitution of the first 4 bounces
           (from successive bounce heights). Values > 1 mean the bounce adds energy.
  fast   - ball hitting the table at 2.5 m/s forward, 3.5 m/s down: vertical restitution
           and how much forward speed it keeps (grip).
  blade  - ball flying at 3 m/s into the right racquet: restitution off the blade.
  pen    - deepest the ball sinks into the table, in mm.
"expect" is the restitution the damping ratio alone would give: exp(-pi*z/sqrt(1-z^2)).
"""
import math

import mujoco
import numpy as np

TABLE_TOP = 0.76
R = 0.02
ZC = TABLE_TOP + R

CANDIDATES = [            # (timestep, timeconst, dampratio)
    (0.0005, 0.004, 0.08),    # original setting: bounces GAIN energy at this timestep
    (0.0002, 0.004, 0.08),
    (0.0005, 0.010, 0.06),
    (0.0005, 0.020, 0.06),
    (0.0005, 0.030, 0.05),    # current pingpong.xml
    (0.0002, 0.010, 0.06),
    (0.0005, 0.020, 0.04),
]


def load(dt, tau, zeta):
    m = mujoco.MjModel.from_xml_path("pingpong.xml")
    m.opt.timestep = dt
    m.geom_solref[:, 0] = tau
    m.geom_solref[:, 1] = zeta
    return m


def ball(m):
    j = m.joint("ball_free")
    return j.qposadr[0], j.dofadr[0]


def start(m, pos, vel):
    d = mujoco.MjData(m)
    q, v = ball(m)
    d.qpos[q:q + 3] = pos
    d.qpos[q + 3:q + 7] = [1, 0, 0, 0]
    d.qvel[v:v + 3] = vel
    mujoco.mj_forward(m, d)
    return d, q, v


def drop_test(m):
    d, q, v = start(m, [0.5, 0.0, ZC + 0.30], [0, 0, 0])
    heights, prev_vz, zmin = [0.30], 0.0, 9.0
    for _ in range(int(4.0 / m.opt.timestep)):
        mujoco.mj_step(m, d)
        z, vz = d.qpos[q + 2], d.qvel[v + 2]
        zmin = min(zmin, z)
        if prev_vz > 0 >= vz:                      # apex after a bounce
            heights.append(z - ZC)
            if len(heights) > 4:
                break
        prev_vz = vz
    es = [math.sqrt(max(h1, 0) / h0) for h0, h1 in zip(heights, heights[1:]) if h0 > 1e-4]
    return es, (ZC - zmin) * 1000


def fast_test(m):
    d, q, v = start(m, [0.3, 0.0, ZC + 0.05], [2.5, 0.0, -3.5])
    vin = None
    for _ in range(int(0.5 / m.opt.timestep)):
        before_z = d.qpos[q + 2]
        before_v = d.qvel[v:v + 3].copy()
        mujoco.mj_step(m, d)
        z = d.qpos[q + 2]
        if vin is None and before_z > ZC + 0.002 >= z:
            vin = before_v
        elif vin is not None and before_z < ZC + 0.002 <= z:
            vout = d.qvel[v:v + 3]
            return vout[2] / -vin[2], vout[0] / vin[0]
    return float("nan"), float("nan")


def blade_test(m):
    face = 1.5 - 0.008 - R                     # ball centre x touching the right blade
    d, q, v = start(m, [1.2, 0.0, 0.95], [3.0, 0.0, 0.0])
    vin = None
    for _ in range(int(0.5 / m.opt.timestep)):
        before_x = d.qpos[q]
        before_v = d.qvel[v:v + 3].copy()
        mujoco.mj_step(m, d)
        x = d.qpos[q]
        if vin is None and before_x < face - 0.002 <= x:
            vin = before_v
        elif vin is not None and before_x > face - 0.002 >= x:
            return -d.qvel[v] / vin[0]
    return float("nan")


def main():
    print(f"mujoco {mujoco.__version__}")
    print(f"{'dt':>7} {'tconst':>7} {'damp':>5} {'expect':>6} | {'drop e (4 bounces)':<27} | "
          f"{'fast e':>6} {'grip':>5} | {'blade e':>7} | {'pen mm':>6}")
    for dt, tau, zeta in CANDIDATES:
        m = load(dt, tau, zeta)
        expect = math.exp(-math.pi * zeta / math.sqrt(1 - zeta * zeta))
        es, pen = drop_test(m)
        fe, grip = fast_test(m)
        be = blade_test(m)
        drop = " ".join(f"{e:.3f}" for e in es)
        print(f"{dt:7.4f} {tau:7.3f} {zeta:5.2f} {expect:6.3f} | {drop:<27} | "
              f"{fe:6.3f} {grip:5.3f} | {be:7.3f} | {pen:6.1f}")


if __name__ == "__main__":
    main()
