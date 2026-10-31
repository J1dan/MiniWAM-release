#!/usr/bin/env python3
"""Execute the configured SAPIEN ray-tracing denoiser on one synthetic frame."""

import numpy as np
import sapien


sapien.render.set_camera_shader_dir("rt")
sapien.render.set_ray_tracing_samples_per_pixel(32)
sapien.render.set_ray_tracing_path_depth(8)
sapien.render.set_ray_tracing_denoiser("oidn")

scene = sapien.Scene()
scene.add_ground(altitude=0)
builder = scene.create_actor_builder()
builder.add_box_visual(half_size=[0.5, 0.5, 0.5], material=[1.0, 0.0, 0.0])
box = builder.build(name="renderer_preflight_box")
box.set_pose(sapien.Pose(p=[0, 0, 0.5]))
scene.set_ambient_light([0.5, 0.5, 0.5])
scene.add_directional_light([0, 1, -1], [0.5, 0.5, 0.5])
mount = scene.create_actor_builder().build_kinematic(name="renderer_preflight_camera")
mount.set_pose(sapien.Pose([-3, 0, 0.5]))
camera = scene.add_mounted_camera(
    "renderer_preflight", mount, sapien.Pose(), 64, 64, 1.0, 0.01, 10
)
scene.update_render()
camera.take_picture()
color = camera.get_picture("Color")
if color.shape != (64, 64, 4) or not np.isfinite(color).all():
    raise RuntimeError(f"Invalid renderer preflight image: {color.shape}")
print(
    f"[MiniWAM] renderer_preflight=passed shape={color.shape} "
    f"range=[{float(color.min()):.6f},{float(color.max()):.6f}]",
    flush=True,
)
