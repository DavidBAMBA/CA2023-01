"""
===============================================================================
Main script 
Creates the Black Hole image
===============================================================================
@author: Eduard Larrañga - 2023
===============================================================================
"""

from common import Image
from config import *
from mpi4py import MPI
import numpy as np


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

#################################### MAIN #####################################

start_time = MPI.Wtime()

detector      = image_plane(D=D, iota = iota, s_side = screen_side, n_pixels = n_pixels)
blackhole     = BlackHole(M)
acc_structure = thin_disk(R_min, R_max)



total_photons = n_pixels * n_pixels
photons_per_rank = total_photons // size 
rest = total_photons % size

# Añade un fotón adicional para los primeros 'rest' procesos
if rank < rest:
    start_idx = rank * (photons_per_rank + 1)
    end_idx = start_idx + photons_per_rank + 1
else:
    start_idx = (rank * photons_per_rank) + rest
    end_idx = start_idx + photons_per_rank


start_alpha, start_beta = divmod(start_idx, len(detector.betaRange))
end_alpha, end_beta = divmod(end_idx, len(detector.betaRange))


image = Image(start_alpha, start_beta, end_alpha, end_beta)


# Photons creation
image.create_photons(blackhole, detector)

# Create the image data
image.create_image(blackhole, acc_structure)

gathered_image_data = comm.gather(image.image_data, root=0)

if rank == 0:
    final_image_data = np.zeros([detector.numPixels, detector.numPixels])
    
    for data in gathered_image_data:
        final_image_data += data

    image.image_data = final_image_data   #final_image_data  

    end_time = MPI.Wtime()
    elapsed_time = end_time - start_time
    print(f"TIME: {elapsed_time} sec")

    #image.plot(savefig=True, filename=filename)
