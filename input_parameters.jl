const working_directory = string("/home/singularity/Dropbox (ASU)/bayesianCodes/julia/illumination_correction/FISH/new_FISH_data/20240327_Ayush_experiment_redo_2/inference/rl_prior_siemens_highSNR/") 
raw_image_size_x::Integer = 512
raw_image_size_y::Integer = 512

# Optical Parameters
const numerical_aperture::Float64 = 1.0
const magnification::Float64 = 100.0
const light_wavelength::Float64 = 0.660# In micrometers
const abbe_diffraction_limit::Float64 = light_wavelength/(2*numerical_aperture) # optical resolution in micrometers
const f_diffraction_limit::Float64 = 1/abbe_diffraction_limit # Diffraction limit in k-space
const sigma::Float64 = sqrt(2.0)/(2.0*pi)*light_wavelength/numerical_aperture # Standard Deviation in PSF

# Camera Parameters
const camera_pixel_size::Float64 = 6.5 # In micrometers
const physical_pixel_size::Float64 = camera_pixel_size/magnification #in micrometers
const dx::Float64 = physical_pixel_size #physical grid spacing
const gain::Float64 = 1.957
const offset::Float64 = 100.0
const noise::Float64 = 2.3
const noise_maps_available = false

# Inference Parameters
const ghost_size::Integer = 16 # Always choose numbers divisible by 4
const half_ghost_size::Integer = ghost_size/2
const covariance_gt::Float64 = 1.0

# Parallelization Parameters
const n_procs_per_dim_x::Integer = 4
const n_procs_per_dim_y::Integer = 4

const total_draws::Integer = 100000
const initial_burn_in_period::Integer = 0	
const annealing_frequency::Integer = total_draws
const annealing_starting_temperature::Float64 = 1.0
const annealing_time_constant::Float64 = 10.0
const annealing_burn_in_period::Integer = 100
const averaging_frequency::Integer = 50

const plotting_frequency::Integer = 10
