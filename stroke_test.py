"""
Headless test of the racquet strokes in pingpong_drex.py: fires balls at each racquet,
runs the same controller as the game (no viewer, no Drex), and prints at the moment the
ball touches the blade what was planned and what MuJoCo actually did.

    python stroke_test.py

Per hit:
  timing     blade contact time minus planned contact time (ms; + = ball arrived late)
  hit at     where on the racquet the ball touched: distance from blade centre, and whether
             MuJoCo's contact normal matches the face normal (an edge hit won't)
  face n     racquet face normal from MuJoCo vs from the script's face_normal()
  tilt/yaw   commanded vs actual joint angles
  swing      planned vs actual racquet x speed (toward the table)
  out        ball velocity leaving the blade: planned / MuJoCo actual / script's bounce
             model fed with the actual pre-contact ball, spin, face normal and racquet speed
  result     where the ball landed
"""
import types

import numpy as np
import mujoco

import pingpong_drex as P

CASES = [            # incoming ball, mirrored for each side: (x0, y0, z0, vx, vy, vz)
    (-0.3, 0.0, 1.00, 3.0, 0.0, 0.6),
    (-0.3, 0.2, 1.05, 4.0, 0.2, 0.0),
    (-0.3, -0.2, 0.95, 3.5, -0.2, 1.0),
    (-0.5, 0.1, 1.10, 2.5, 0.1, 1.2),
    (-0.2, -0.3, 1.00, 3.2, -0.1, -0.5),
]


def fmt(v):
    return "(" + ",".join(f"{x:+.2f}" for x in v) + ")"


def main():
    args = types.SimpleNamespace(no_drex=True, max_age=0.6, quiet=True, serve_speed=2.0,
                                 sim_speed=1.0)
    g = P.Game(args)
    m, d = g.model, g.data
    ball = m.geom("ball_geom").id
    names = {+1: "right", -1: "left"}
    blade = {s: m.geom(f"{n}_blade").id for s, n in names.items()}
    body = {s: m.body(f"{n}_racquet").id for s, n in names.items()}
    jq = {s: {k: m.joint(f"{n}_{k}").qposadr[0] for k in ("x", "y", "z", "yaw", "tilt")}
          for s, n in names.items()}
    jv = {s: {k: m.joint(f"{n}_{k}").dofadr[0] for k in ("x", "y", "z")}
          for s, n in names.items()}
    table = m.geom("table_top").id
    net = m.geom("net").id
    steps_per_frame = int(round(1 / 60 / m.opt.timestep))

    def touching(gid):
        for c in d.contact[:d.ncon]:
            if ball in (c.geom1, c.geom2) and gid in (c.geom1, c.geom2):
                n = np.array(c.frame[:3])
                return n if c.geom1 == ball else -n      # normal pointing away from the ball
        return None

    for side in (+1, -1):
        for i, (x0, y0, z0, vx, vy, vz) in enumerate(CASES):
            mujoco.mj_resetData(m, d)
            q, v = g.bq, g.bv
            d.qpos[q:q + 3] = [side * x0, y0, z0]
            d.qpos[q + 3:q + 7] = [1, 0, 0, 0]
            d.qvel[v:v + 3] = [side * vx, vy, vz]
            mujoco.mj_forward(m, d)
            g.shot += 1
            g.last_dir = side
            g.plans = {-1: None, +1: None}
            g.guess = {-1: None, +1: None}
            g.aims = {-1: None, +1: None}
            g.decisions = {}

            info, prev_v, prev_w, landed = None, None, None, None
            for step in range(int(2.5 / m.opt.timestep)):
                if step % steps_per_frame == 0 and info is None:
                    g.control(-1)
                    g.control(+1)
                g.drive_x()
                prev_v = d.qvel[v:v + 3].copy()
                prev_w = g.ball_spin().copy()
                prev_p = d.qpos[q:q + 3].copy()
                mujoco.mj_step(m, d)
                if info is None:
                    cn = touching(blade[side])
                    if cn is not None:
                        aim = g.aims[side]
                        R = d.xmat[body[side]].reshape(3, 3)
                        n_mj = R @ np.array([-side, 0.0, 0.0])
                        rvel = np.array([d.qvel[jv[side]["x"]], d.qvel[jv[side]["y"]],
                                         d.qvel[jv[side]["z"]]])
                        centre = d.xpos[body[side]]
                        info = dict(t=d.time, aim=aim, n_mj=n_mj, cn=cn, vin=prev_v, w=prev_w,
                                    rvel=rvel, pin=prev_p,
                                    off=np.linalg.norm((prev_p - centre) - ((prev_p - centre) @ n_mj) * n_mj),
                                    tilt=d.qpos[jq[side]["tilt"]], yaw=d.qpos[jq[side]["yaw"]])
                elif "vout" not in info:
                    if touching(blade[side]) is None:
                        info["vout"] = d.qvel[v:v + 3].copy()
                else:
                    if touching(net) is not None:
                        landed = "net"
                        break
                    if touching(table) is not None:
                        x, y = d.qpos[q], d.qpos[q + 1]
                        landed = ("own side" if x * side > 0 else "landed") + f" at x={x:+.2f} y={y:+.2f}"
                        break
                    if d.qpos[q + 2] < 0.3:
                        landed = f"off the table (x={d.qpos[q]:+.2f} y={d.qpos[q + 1]:+.2f})"
                        break
            tag = f"{names[side][0].upper()}{i + 1}"
            if info is None:
                bx, by, bz = d.qpos[q:q + 3]
                print(f"{tag}: NO CONTACT with the {names[side]} blade "
                      f"(aims={g.aims[side] is not None}, ball ended at x={bx:+.2f} z={bz:+.2f})")
                continue
            a = info["aim"]
            if a is None:
                print(f"{tag}: contact but no stroke plan was made")
                continue
            model_out = P.blade_hit(info["vin"], info["w"], info["n_mj"], info["rvel"])
            print(f"{tag}: timing {1000 * (info['t'] - a['t_c']):+5.0f} ms | hit {100 * info['off']:.1f} cm "
                  f"from centre, contact-normal.face={-info['cn'] @ info['n_mj']:+.2f} | "
                  f"face n mj {fmt(info['n_mj'])} script {fmt(a['normal'])}")
            print(f"     tilt cmd {a['tilt']:+.2f} act {info['tilt']:+.2f}  yaw cmd {a['yaw']:+.2f} "
                  f"act {info['yaw']:+.2f} | swing plan {a['u']:.2f} act {-side * info['rvel'][0]:.2f} m/s"
                  f" (racquet y,z vel {info['rvel'][1]:+.2f},{info['rvel'][2]:+.2f})")
            print(f"     ball in {fmt(info['vin'])} spin {fmt(info['w'])}")
            print(f"     out planned {fmt(a['v_target'])}  MuJoCo {fmt(info.get('vout', [np.nan] * 3))}"
                  f"  model(actual inputs) {fmt(model_out)}  -> {landed}")


if __name__ == "__main__":
    main()
