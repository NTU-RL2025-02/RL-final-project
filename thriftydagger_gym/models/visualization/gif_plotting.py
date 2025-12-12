import trajectory_plotting
from PIL import Image
from pathlib import Path


EPOCHS=46
pngs = []
for epoch in range(1, EPOCHS):
    trajectory_plotting.plot(
        input_type='hdf5', 
        input_path=Path(f'hdf5/epoch{epoch}_trajectories.hdf5'), 
        output_path=Path(f'png/epoch{epoch}_trajectories.png'), 
        maze_layout='four_rooms', 
        sample_amount=None, 
        hdf5_training_traj=True,
        hdf5_testing_traj=False
    )
    pngs.append(f'png/epoch{epoch}_trajectories.png')


# Load all PNG files (sorted by filename)
frames = [Image.open(img) for img in pngs]

# Save as GIF
frames[0].save(
    "output.gif",
    save_all=True,
    append_images=frames[1:],
    duration=1000,   # milliseconds per frame
    loop=0          # loop forever
)
