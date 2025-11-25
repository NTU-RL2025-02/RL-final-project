"""
Quick utility to inspect robosuite observation and action spaces.
Run: python thrifty/utils/test.py
"""

import numpy as np
import robosuite as suite


def make_env(env_name="NutAssembly"):
    # Match the controller config used in run_thriftydagger.py (BASIC -> OSC_POSE)
    controller_config = {
        "type": "BASIC",
        "body_parts": {
            "right": {
                "type": "OSC_POSE",
                "input_max": 1,
                "input_min": -1,
                "output_max": [0.05, 0.05, 0.05, 0.5, 0.5, 0.5],
                "output_min": [-0.05, -0.05, -0.05, -0.5, -0.5, -0.5],
                "kp": 150,
                "damping_ratio": 1,
                "impedance_mode": "fixed",
                "kp_limits": [0, 300],
                "damping_ratio_limits": [0, 10],
                "position_limits": None,
                "orientation_limits": None,
                "uncouple_pos_ori": True,
                "input_type": "delta",
                "input_ref_frame": "base",
                "interpolation": None,
                "ramp_ratio": 0.2,
                "gripper": {"type": "GRIP"},
            }
        },
    }

    env = suite.make(
        env_name=env_name,
        robots="Panda",
        controller_configs=controller_config,
        has_renderer=False,
        has_offscreen_renderer=False,
        render_camera="agentview",
        ignore_done=True,
        use_camera_obs=False,
        use_object_obs=True,
        reward_shaping=True,
        control_freq=20,
        hard_reset=True,
    )
    return env


def main():
    env = make_env()
    obs = env.reset()
    # unwrap gymnasium-style (obs, info)
    if isinstance(obs, tuple):
        obs = obs[0]

    print("=== Action Space ===")
    if hasattr(env, "action_space"):
        print(env.action_space)
    print(f"Action Dimension: {env.action_dim}")
    lo, hi = env.action_spec
    print(f"Action Spec Low : {lo}")
    print(f"Action Spec High: {hi}")

    print("\n=== Observation ===")
    if isinstance(obs, dict):
        total_dim = 0
        for key, val in obs.items():
            arr = np.asarray(val)
            flat_dim = arr.size
            total_dim += flat_dim
            preview = arr.flatten()[:6]
            print(f"{key:25} shape={arr.shape} dim={flat_dim:3d} preview={preview}")
        print(f"\nTotal flattened obs dim: {total_dim}")
        if "object-state" in obs:
            print(f"object-state length: {np.asarray(obs['object-state']).size}")
            obj = np.asarray(obs["object-state"]).flatten()
            print("object-state (all values):")
            print(obj)
    else:
        arr = np.asarray(obs)
        print(f"Obs shape: {arr.shape}")
        print("First 10 vals:", arr.flatten()[:10])


if __name__ == "__main__":
    main()
