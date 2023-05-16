import taichi as ti
import argparse
import os
from FLIP import FLIP_2D
import imageio
import numpy as np

ti.init(arch=ti.cpu, default_ip=ti.i32)

obstacle = ti.Vector.field(2, dtype=ti.f32, shape=1)
env = FLIP_2D(1.0 / 120.0, 100, 100, 1.0, 1.0, 35 * 35)

obstacle[0][0] = 0.7
obstacle[0][1] = 0.25
obstacleRadius = 0.15

@ti.kernel
def setup_scene():
    for i, j in ti.ndrange(env.grid_x, env.grid_y):
        if i < 35 and j < 35:
            env.pos[i *35 + j] = ti.Vector([0.1 + 4 * env.radius * i, 0.9 - 4 * env.radius * j])
        env.status[i * 100 + j] = 1.0
        if (i == 0 or i == env.grid_x - 1 or j ==0):
            env.status[i * 100 + j] = 0.0
        xx = (i + 0.5) * env.dx
        yy = (j + 0.5) * env.dx
        if (xx - 0.7)**2 + (yy - 0.25)**2 < obstacleRadius**2:
            env.status[i * 100 + j] = 0.0

def main(args):

    window = ti.ui.Window("Taichi Paper Simulation on GGUI", (960, 960),
                          vsync=True, show_window=False)
    canvas = window.get_canvas()
    canvas.set_background_color((1, 1, 1))
    setup_scene()

    tot_step = 0
    frames = []
    save_path = f"imgs/FLIP"
    canvas.set_background_color((1, 1, 1))
    canvas.circles(env.pos, env.radius, (0, 0, 1))
    canvas.circles(obstacle, obstacleRadius, (0.1, 0.1, 0.1))
    canvas.set_image(window.get_image_buffer_as_numpy())
    window.save_image(os.path.join(save_path, f"{0}.png"))
    while window.running:
        tot_step += 1
        if(tot_step > 20):
            break
        print(tot_step)

        env.simulate_step(0.95, 2, 50, 1.9, obstacle[0][0], obstacle[0][1], obstacleRadius)
        canvas.set_background_color((1, 1, 1))
        canvas.circles(env.pos, env.radius, (0, 0, 1))
        canvas.circles(obstacle, obstacleRadius, (0.1, 0.1, 0.1))
        canvas.set_image(window.get_image_buffer_as_numpy())
        window.save_image(os.path.join(save_path, f"{tot_step}.png"))

    for i in range(1, tot_step):
        filename = os.path.join(save_path, f"{i}.png")
        frames.append(imageio.imread(filename))

    gif_name = os.path.join(save_path, f"GIF_{args.exp_id}.gif")
    imageio.mimsave(gif_name, frames, 'GIF', duration=0.02)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_id", type=int, default=0)
    main(parser.parse_args())
