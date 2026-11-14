import mujoco, numpy as np
m = mujoco.MjModel.from_xml_path("pingpong.xml")
d = mujoco.MjData(m)
j = m.joint("ball_free").qposadr[0]
v = m.joint("ball_free").dofadr[0]
d.qpos[j:j+3] = [0.5, 0, 1.18]
for i in range(1200):
    mujoco.mj_step(m, d)
    if i % 20 == 0:
        print(f"t={d.time:.3f}  z={d.qpos[j+2]:.3f}  x={d.qpos[j]:.2f}  speed={np.linalg.norm(d.qvel[v:v+3]):.2f}")