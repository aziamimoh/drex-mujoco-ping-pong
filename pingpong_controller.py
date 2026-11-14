import time
import numpy as np
import mujoco
import mujoco.viewer

model = mujoco.MjModel.from_xml_path("pingpong.xml")
data = mujoco.MjData(model)

ball_qpos = model.joint("ball_free").qposadr[0]
ball_qvel = model.joint("ball_free").dofadr[0]


def act(name):
    return model.actuator(name).id


LEFT = dict(y=act("left_y_ctrl"), z=act("left_z_ctrl"), tilt=act("left_tilt_ctrl"))
RIGHT = dict(y=act("right_y_ctrl"), z=act("right_z_ctrl"), tilt=act("right_tilt_ctrl"))

HOME_Z = 0.95          # racquet rest height (matches the XML)
RACQUET_X = 1.5        # racquet distance from the table center
RESERVE_AFTER = 15.0   # seconds before forcing a re-serve
MAX_BALL_SPEED = 8.0   # m/s safety cap
rng = np.random.default_rng()


def serve():
    side = rng.choice([-1, 1])
    data.qpos[ball_qpos:ball_qpos + 3] = [-side * 1.0, rng.uniform(-0.3, 0.3), 1.15]
    data.qpos[ball_qpos + 3:ball_qpos + 7] = [1, 0, 0, 0]
    data.qvel[ball_qvel:ball_qvel + 3] = [side * 2.0, rng.uniform(-0.3, 0.3), 0.0]
    data.qvel[ball_qvel + 3:ball_qvel + 6] = 0
    mujoco.mj_forward(model, data)


def predict_y_at_x(pos, vel, x_target):
    """Straight-line prediction of the ball's y when it reaches x_target."""
    if abs(vel[0]) < 1e-3:
        return pos[1]
    t = (x_target - pos[0]) / vel[0]
    return pos[1] + vel[1] * t if t > 0 else pos[1]


def control(paddle, x_side):
    pos = data.qpos[ball_qpos:ball_qpos + 3]
    vel = data.qvel[ball_qvel:ball_qvel + 3]
    incoming = (vel[0] * x_side) > 0 and (pos[0] * x_side) > 0.1
    if incoming:
        target_y = np.clip(predict_y_at_x(pos, vel, x_side * RACQUET_X), -0.9, 0.9)
        target_z = np.clip(pos[2] - HOME_Z, -0.2, 0.4)
        near = abs(pos[0] - x_side * RACQUET_X) < 0.35
        tilt = -x_side * 0.3 if near else 0.0   # flip the sign if returns go the wrong way
    else:
        target_y, target_z, tilt = 0.0, 0.0, 0.0
    data.ctrl[paddle["y"]] = target_y
    data.ctrl[paddle["z"]] = target_z
    data.ctrl[paddle["tilt"]] = tilt


def reset_and_serve():
    mujoco.mj_resetData(model, data)
    serve()


reset_and_serve()
last_serve = time.time()
steps_per_frame = int(1 / 60 / model.opt.timestep)

with mujoco.viewer.launch_passive(model, data) as viewer:
    viewer.cam.lookat[:] = [0, 0, 0.8]
    viewer.cam.azimuth, viewer.cam.elevation, viewer.cam.distance = 90, -20, 4.5

    while viewer.is_running():
        t0 = time.time()

        control(LEFT, -1)
        control(RIGHT, +1)

        for _ in range(steps_per_frame):
            mujoco.mj_step(model, data)

        # --- safety checks on the ball ---
        vel = data.qvel[ball_qvel:ball_qvel + 3]
        speed = np.linalg.norm(vel)
        if not np.isfinite(speed) or not np.all(np.isfinite(data.qpos)):
            print("simulation went NaN, resetting")
            reset_and_serve()
            last_serve = time.time()
        elif speed > MAX_BALL_SPEED:
            print(f"ball too fast: {speed:.1f} m/s at {data.qpos[ball_qpos:ball_qpos + 3]}")
            vel *= MAX_BALL_SPEED / speed

        p = data.qpos[ball_qpos:ball_qpos + 3]
        if (p[2] < 0.1 or abs(p[0]) > 2.5 or abs(p[1]) > 1.5
                or time.time() - last_serve > RESERVE_AFTER):
            serve()
            last_serve = time.time()

        viewer.sync()
        time.sleep(max(0, 1 / 60 - (time.time() - t0)))