include("input_parameters.jl")

using Distributed

println("Adding processors...")
flush(stdout);
addprocs(n_procs_per_dim_x*n_procs_per_dim_y, topology=:master_worker)
println("Done.")
flush(stdout);

@everywhere using Random, Distributions

@everywhere using LinearAlgebra
@everywhere using SpecialFunctions
@everywhere using FFTW

using HDF5
using TiffImages
using Plots

@everywhere workers() include("input_parameters.jl")
function get_camera_calibration_data()

    local offset_map::Matrix{Float64}
    local variance_map::Matrix{Float64}
    local error_map::Matrix{Float64}
    local gain_map::Matrix{Float64}


    if noise_maps_available == true
        file_name = string(working_directory,
                            "offset_map.tif")
        offset_map = TiffImages.load(file_name)
        offset_map = Float64.(offset_map)

        file_name = string(working_directory,
                            "variance_map.tif")
        variance_map = TiffImages.load(file_name)
        variance_map = Float64.(variance_map)
		error_map = sqrt.(variance_map)

        file_name = string(working_directory,
                            "gain_map.tif")
        gain_map = TiffImages.load(file_name)
        gain_map = Float64.(gain_map)
    else

        offset_map = offset .* ones(raw_image_size_x, raw_image_size_y)
        gain_map = gain .* ones(raw_image_size_x, raw_image_size_y)
        error_map = noise .* ones(raw_image_size_x, raw_image_size_y)

    end

    return offset_map, error_map, gain_map
end

println("Importing camera calibration data...")
flush(stdout);
const offset_map, error_map, gain_map = get_camera_calibration_data()
println("Done.")
println("median gain = ", median(gain_map))
println("median offset = ", median(offset_map))
println("median error = ", median(error_map))
flush(stdout);

function get_input_raw_image()
 	file_name = string(working_directory,
				"raw_image.tif")
   	img = TiffImages.load(file_name)
#   	img = reinterpret(UInt16, img)
   	img = Float64.(img)

    return img
end

println("Importing raw image...")
flush(stdout);
const input_raw_image::Matrix{Float64} = get_input_raw_image()
println("Done.")
println("Size of Raw Image = ", size(input_raw_image))
println("median ADU in raw_image = ", median(input_raw_image))
println("maximum ADU in raw_image = ", maximum(input_raw_image))
println("minimum ADU in raw_image = ", minimum(input_raw_image))



flush(stdout);

@everywhere const img_size_x = size($input_raw_image)[1]
@everywhere const img_size_y = size($input_raw_image)[2]


function get_camera_calibration_data_with_ghosts(
                input_map::Matrix{Float64}, average_val)

	img::Matrix{Float64} = average_val .* 
                ones(img_size_x+2*ghost_size, img_size_y+2*ghost_size)
	img[ghost_size+1:end-ghost_size,
			ghost_size+1:end-ghost_size] = input_map[1:end, 1:end]
	return img
end

println("Adding ghosts to calibration data...")
flush(stdout);

const gain_map_with_ghosts = 
            get_camera_calibration_data_with_ghosts(gain_map, gain)
println("Size of gain map With ghosts = ", size(gain_map_with_ghosts))

const offset_map_with_ghosts = 
            get_camera_calibration_data_with_ghosts(offset_map, offset)
println("Size of offset map With ghosts = ", size(offset_map_with_ghosts))

const error_map_with_ghosts = 
            get_camera_calibration_data_with_ghosts(error_map, noise)
println("Size of error map With ghosts = ", size(error_map_with_ghosts))

println("Done.")
flush(stdout);



function get_raw_image_with_ghosts(input_raw_img::Matrix{Float64})
	img = Matrix{Float64}[]
	img = zeros(img_size_x+2*ghost_size, img_size_y+2*ghost_size)
	img[ghost_size+1:end-ghost_size,
				ghost_size+1:end-ghost_size] =
					input_raw_img[:, :]
	return img
end

println("Adding ghosts to raw image...")
flush(stdout);
const raw_image_with_ghosts = get_raw_image_with_ghosts(input_raw_image)
println("Done.")
println("Size of Raw Image With Ghosts = ", size(raw_image_with_ghosts[1]))
flush(stdout);

function get_flatfield()

#    file_name = string(working_directory,
#                            "flatfield.tif")
#   	flatfield_image = TiffImages.load(file_name)
#   	flatfield_image = Float64.(flatfield_image)
#	flatfield_image = flatfield_image ./ maximum(flatfield_image)
	flatfield_image = ones(raw_image_size_x, raw_image_size_y)


	return flatfield_image
end


println("Importing illumination profile...")
flush(stdout);
const flatfield_without_ghosts::Matrix{Float64} = get_flatfield()
println("Done.")
println("Size of Illumination Profile = ",
				size(flatfield_without_ghosts))
flush(stdout);

function get_flatfield_with_ghosts(
					input_illum_profile::Matrix{Float64})
    illum_profile = zeros(img_size_x+2*ghost_size, img_size_y+2*ghost_size)
    illum_profile[ghost_size+1:end-ghost_size,
			ghost_size+1:end-ghost_size] = input_illum_profile[:, :]
	return illum_profile
end

println("Adding ghosts to illumination Profile...")
flush(stdout);
const flatfield = get_flatfield_with_ghosts(
								flatfield_without_ghosts)
println("Done.")
println("Size of Illumination Profile With Ghosts = ",
				size(flatfield))
flush(stdout);

const grid_physical_1D_x = dx .* collect(-(raw_image_size_x/2 + ghost_size):
					(img_size_x/2 + ghost_size - 1)) # in micrometers
const grid_physical_1D_y = dx .* collect(-(img_size_y/2 + ghost_size):
					(img_size_y/2 + ghost_size - 1)) # in micrometers


@everywhere function incoherent_PSF(x_c::Vector{Float64}, x_e::Vector{Float64})
	return exp(-norm(x_c-x_e)^2/(2.0*sigma^2)) /
					(sqrt(2.0*pi) * sigma)^(size(x_e)[1])
end

println("Done.")
flush(stdout);

function FFT_incoherent_PSF()
	psf_on_grid = zeros(img_size_x+2*ghost_size, img_size_y+2*ghost_size)
	for i in 1:img_size_x + ghost_size
		for j in 1:img_size_y + ghost_size
			x_e::Vector{Float64} = [grid_physical_1D_x[i], grid_physical_1D_y[j]]
			psf_on_grid[i, j] =  incoherent_PSF([0.0, 0.0], x_e)
		end
	end
	return fft(ifftshift(psf_on_grid))
end


println("Computing FFT of the PSF...")
flush(stdout);
const FFT_point_spread_function =  FFT_incoherent_PSF()
println("Done.")
flush(stdout);

const modulation_transfer_function = abs.(fftshift(FFT_point_spread_function))[ghost_size+1:end-ghost_size, ghost_size+1:end-ghost_size] 
const modulation_transfer_function_vectorized = vec(modulation_transfer_function) ./ sum(modulation_transfer_function)

function get_widefield_image(illumination_prof::Matrix{Float64},
					fluorophore_density::Matrix{Float64})
	illuminated_density::Matrix{Float64} =
					(illumination_prof .* fluorophore_density)
	FFT_illuminated_density::Matrix{ComplexF64} =
				fft(ifftshift(illuminated_density)) .* dx^2
	FFT_final::Matrix{ComplexF64} =
				FFT_point_spread_function .* FFT_illuminated_density

 	image::Matrix{Float64} = abs.(real.(fftshift(ifft(FFT_final))))

	return image
end

function get_mean_image(ground_truth::Matrix{Float64})
	final_image::Matrix{Float64} =
		get_widefield_image(flatfield, ground_truth)

	val_range_x = collect(ghost_size+1:1:ghost_size+img_size_x)
	val_range_y = collect(ghost_size+1:1:ghost_size+img_size_y)
	mod_fft_image = vec(abs.(fftshift(fft(ifftshift(final_image))))[val_range_x, val_range_y]) .+ eps()
	log_prior::Float64 = logpdf(Dirichlet(modulation_transfer_function_vectorized), 
									mod_fft_image ./ sum(mod_fft_image))
	return final_image, log_prior
end


@everywhere workers() begin
	const i_procs::Integer = (myid()-2) % n_procs_per_dim_x
	const j_procs::Integer = (myid()-2 - i_procs)/n_procs_per_dim_x

	const im_raw::Integer = i_procs*img_size_x/n_procs_per_dim_x + 1
	const ip_raw::Integer = 2*ghost_size + (i_procs+1)*img_size_x/n_procs_per_dim_x
	const jm_raw::Integer = j_procs*img_size_y/n_procs_per_dim_y + 1
	const jp_raw::Integer = 2*ghost_size + (j_procs+1)*img_size_y/n_procs_per_dim_y
	const sub_size_raw_x::Integer = img_size_x/n_procs_per_dim_x + 2*ghost_size
	const sub_size_raw_y::Integer = img_size_y/n_procs_per_dim_y + 2*ghost_size
end

@everywhere workers() function get_sub_image(img::Matrix{Float64},
				im::Integer, ip::Integer, jm::Integer, jp::Integer)
	sub_img = img[im:ip, jm:jp]
	return sub_img
end

println("Assigning sections of raw image to each processor...")
flush(stdout);
@everywhere workers() const sub_raw_image = 
                    get_sub_image($raw_image_with_ghosts,
				im_raw, ip_raw, jm_raw, jp_raw)
println("Done.")
flush(stdout);

@everywhere workers() function get_sub_calibration_map(
                full_map::Matrix{Float64},
				im::Integer, ip::Integer, jm::Integer, jp::Integer)
	sub_map::Matrix{Float64} = full_map[im:ip, jm:jp]
	return sub_map
end

println("Assigning sections of calibration maps to each processor...")
flush(stdout);
@everywhere workers() const sub_gain_map = 
                    get_sub_calibration_map($gain_map_with_ghosts,
					im_raw, ip_raw, jm_raw, jp_raw)
@everywhere workers() const sub_offset_map = 
                    get_sub_calibration_map($offset_map_with_ghosts,
					im_raw, ip_raw, jm_raw, jp_raw)
@everywhere workers() const sub_error_map = 
                    get_sub_calibration_map($error_map_with_ghosts,
					im_raw, ip_raw, jm_raw, jp_raw)

println("Done.")
flush(stdout);

println("Assigning sections of illumination profile to each processor...")
flush(stdout);
@everywhere workers() const sub_flatfield =
					get_sub_image($flatfield,
						im_raw, ip_raw, jm_raw, jp_raw)
println("Done.")
flush(stdout);



################Inference Part#######################################
@everywhere workers() const grid_physical_1D_ij =
			dx .* collect(-ghost_size:ghost_size) # in micrometers

@everywhere workers() function FFT_incoherent_PSF_ij()
	psf_on_grid = zeros(2*ghost_size+1, 2*ghost_size+1)
	for i in 1:2*ghost_size+1
		for j in 1:2*ghost_size+1
			x_e::Vector{Float64} = [grid_physical_1D_ij[i],
											grid_physical_1D_ij[j]]
			psf_on_grid[i, j] =  incoherent_PSF([0.0, 0.0], x_e)
		end
	end
	return fft(ifftshift(psf_on_grid))
end

println("Computing FFT of the PSF in neighborhood for inference...")
flush(stdout);
@everywhere workers() const FFT_point_spread_function_ij =
											FFT_incoherent_PSF_ij()
println("Done.")
flush(stdout);

@everywhere workers() modulation_transfer_function_ij = abs.(fftshift(FFT_point_spread_function_ij)) 
@everywhere workers() modulation_transfer_function_ij_vectorized = vec(modulation_transfer_function_ij) ./ sum(modulation_transfer_function_ij)

@everywhere workers() function get_widefield_image_ij(
						illumination_prof::Matrix{Float64},
						fluorophore_density::Matrix{Float64},
						i::Integer, j::Integer)

  	i_minus::Integer = i-ghost_size
 	i_plus::Integer = i+ghost_size
 	j_minus::Integer = j-ghost_size
 	j_plus::Integer = j+ghost_size

	im::Integer = half_ghost_size + 1
	ip::Integer = half_ghost_size + 1 + ghost_size

	jm::Integer = half_ghost_size + 1
	jp::Integer = half_ghost_size + 1 + ghost_size



	illuminated_density::Matrix{Float64} = (illumination_prof[i_minus:i_plus,
								j_minus:j_plus] .*
							fluorophore_density[i_minus:i_plus,
								j_minus:j_plus])
	FFT_illuminated_density::Matrix{ComplexF64} =
					fft(ifftshift(illuminated_density)) .* dx^2
	FFT_final::Matrix{ComplexF64} =
				FFT_point_spread_function_ij .* FFT_illuminated_density

 	image::Matrix{Float64} = abs.(real.(fftshift(ifft(FFT_final))))

 	return image
end


@everywhere workers() function get_mean_image_ij(
                            ground_truth::Matrix{Float64},
					i::Integer, j::Integer)                                 
	im::Integer = half_ghost_size + 1
	ip::Integer = half_ghost_size + 1 + ghost_size

	jm::Integer = half_ghost_size + 1
	jp::Integer = half_ghost_size + 1 + ghost_size

	final_image::Matrix{Float64} =
		get_widefield_image_ij(sub_flatfield,
							ground_truth, i, j)
	mod_fft_image = vec(abs.(fftshift(fft(ifftshift(final_image))))) .+ eps()
	log_prior::Float64 = logpdf(Dirichlet(modulation_transfer_function_ij_vectorized), 
								mod_fft_image ./ sum(mod_fft_image))

    return final_image[im:ip, jm:jp], log_prior
end

@everywhere workers() function get_shot_noise_image_ij(
                                ii::Integer, jj::Integer,
						shot_noise_image::Matrix{Float64})

	i_minus::Integer = ii - half_ghost_size
	i_plus::Integer = ii + half_ghost_size
	j_minus::Integer = jj - half_ghost_size
	j_plus::Integer = jj + half_ghost_size

 	shot_noise_img_ij = shot_noise_image[i_minus:i_plus,
						j_minus:j_plus]
	return shot_noise_img_ij
end

@everywhere workers() function get_log_likelihood_ij(
					ii::Integer, jj::Integer,
					mean_img_ij::Matrix{Float64},
					shot_noise_img_ij::Matrix{Float64})

	i_minus::Integer = 1
	i_plus::Integer = ghost_size+1

	if (i_procs== 0) && (0 < ii - ghost_size <=
						Integer(half_ghost_size))

		i_minus += (half_ghost_size - (ii - ghost_size - 1))

	end
	if (i_procs==n_procs_per_dim_x-1) &&
			(0 <= (sub_size_raw_x - ghost_size) - ii < 
			 			Integer(half_ghost_size))

		i_plus -= (half_ghost_size + ii - (sub_size_raw_x - ghost_size))

	end

	j_minus::Integer = 1
	j_plus::Integer = ghost_size+1

	if (j_procs== 0) && (0 < jj - ghost_size <=
						Integer(half_ghost_size))

		j_minus += (half_ghost_size - (jj - ghost_size - 1))

	end
	if (j_procs==n_procs_per_dim_y-1) &&
			(0 <= (sub_size_raw_y - ghost_size) - jj < 
			 			Integer(half_ghost_size))

		j_plus -= (half_ghost_size + jj - (sub_size_raw_y - ghost_size))

	end



	log_likelihood_ij::Float64 = 0.0
	log_likelihood_ij += sum(logpdf.(
				Poisson.(mean_img_ij[i_minus:i_plus,
							j_minus:j_plus]),
				shot_noise_img_ij[i_minus:i_plus,
							j_minus:j_plus]))

	return log_likelihood_ij
end


# The following Gibbs sampler computes likelihood for the neighborhood only.
@everywhere workers() function sample_gt_neighborhood(
					temperature::Float64,
					gt::Matrix{Float64},
					shot_noise_image::Matrix{Float64})

	local ii::Integer
	local jj::Integer
	local shot_noise_img_ij::Matrix{Float64}
	local mean_img_ij::Matrix{Float64}
	local old_log_likelihood::Float64
	local old_log_prior::Float64
	local old_jac::Float64
	local old_log_posterior::Float64
	local proposed_mean_img_ij::Matrix{Float64}
	local new_log_likelihood::Float64
	local new_log_prior::Float64
	local new_jac::Float64
	local new_log_posterior::Float64
	local log_hastings::Float64
	local expected_photon_count::Float64
	local proposed_shot_noise_pixel::Float64
	local log_forward_proposal_probability::Float64
	local log_backward_proposal_probability::Float64

	n_accepted::Integer = 0

	for i in collect(ghost_size+1:sub_size_raw_x-ghost_size)
		for j in collect(ghost_size+1:sub_size_raw_y-ghost_size)

			shot_noise_img_ij = get_shot_noise_image_ij(i, j,
							shot_noise_image)

			mean_img_ij, old_log_prior = get_mean_image_ij(gt, i, j)

			old_log_likelihood = get_log_likelihood_ij(i, j,
							mean_img_ij, shot_noise_img_ij)

			old_jac = log(gt[i, j])
			old_log_posterior = old_log_likelihood + old_log_prior

			proposed_gt = copy(gt)
			proposed_gt[i, j] =
					rand(Normal(log(gt[i, j]), 
									covariance_gt), 1)[1]
		    	proposed_gt[i, j] = exp.(proposed_gt[i, j])
			proposed_mean_img_ij, new_log_prior = get_mean_image_ij(proposed_gt, i, j)

			new_log_likelihood = get_log_likelihood_ij(i, j,
							proposed_mean_img_ij, shot_noise_img_ij)

			new_jac = log(proposed_gt[i, j])
			new_log_posterior = new_log_likelihood + new_log_prior

			log_hastings = (1.0/temperature) *
                        (new_log_posterior - old_log_posterior) +
							new_jac - old_jac



			if log_hastings > log(rand())
				gt = copy(proposed_gt)
				mean_img_ij = copy(proposed_mean_img_ij)
				n_accepted += 1
			end



#			# Sample Intermediate Expected Photon Counts on each Pixel
#			# Choose the central pixel in the mean image
#			# for expected photon count
#
# 			expected_photon_count = mean_img_ij[
# 							Integer(half_ghost_size+1), 
# 							Integer(half_ghost_size+1)]
# 
#   			old_log_likelihood = logpdf(Normal(sub_gain_map[i, j]*
#                            	shot_noise_image[i, j] +
#   							sub_offset_map[i, j],
#                             sub_error_map[i, j] .+ eps()),
#  							sub_raw_image[i, j])
# 
#   			old_log_prior = logpdf(Poisson(expected_photon_count),
#   								shot_noise_image[i, j])
# 
#    			old_log_posterior = old_log_likelihood  + old_log_prior
#  
#   			proposed_shot_noise_pixel =
#  					rand(Poisson(shot_noise_image[i, j]), 1)[1]
# 
#   			new_log_likelihood = logpdf(Normal(sub_gain_map[i, j]*
#                               	proposed_shot_noise_pixel +
#   							  	sub_offset_map[i, j], 
#  								sub_error_map[i, j] .+ eps()),
#  								sub_raw_image[i, j])
# 
#   			new_log_prior = logpdf(Poisson(expected_photon_count),
#   								proposed_shot_noise_pixel)
# 
#     			new_log_posterior = new_log_likelihood  + new_log_prior
# 
# 			log_forward_proposal_probability = 
# 						logpdf(Poisson(shot_noise_image[i, j]),
#   									proposed_shot_noise_pixel)
# 
# 			log_backward_proposal_probability = 
# 						logpdf(Poisson(proposed_shot_noise_pixel),
#   									shot_noise_image[i, j])
#  
#   			log_hastings = (1.0/temperature)*
#                               	(new_log_posterior - old_log_posterior) +
# 								log_backward_proposal_probability -
# 								log_forward_proposal_probability
#  
#   			if log_hastings > log(rand())
#   				shot_noise_image[i, j] =
#                                      proposed_shot_noise_pixel
#   			end

		end
	end
	return gt, shot_noise_image, n_accepted
end

function get_log_likelihood(ground_truth::Matrix{Float64},
							shot_noise_image::Matrix{Float64})

	log_likelihood::Float64 = 0.0

	mean_image::Matrix{Float64}, log_prior::Float64 = get_mean_image(ground_truth)
	val_range_x = collect(ghost_size+1:1:ghost_size+img_size_x)
	val_range_y = collect(ghost_size+1:1:ghost_size+img_size_y)


	log_likelihood += sum(logpdf.(Poisson.(
				mean_image[val_range_x, val_range_y] .+ eps()),
				shot_noise_image[val_range_x, val_range_y]))
	log_likelihood += sum(logpdf.(Normal.(
                (gain_map_with_ghosts[val_range_x, val_range_y] .*
				shot_noise_image[val_range_x, val_range_y]) .+
					offset_map_with_ghosts[val_range_x, val_range_y],
                    error_map_with_ghosts[val_range_x, val_range_y] .+ eps()),
					raw_image_with_ghosts[val_range_x, val_range_y]))

	return log_likelihood, log_prior
end

function compute_full_log_posterior(gt::Matrix{Float64},
							shot_noise_image::Matrix{Float64})

	log_likelihood::Float64, log_prior::Float64 = get_log_likelihood(gt, shot_noise_image)
	log_posterior::Float64 = log_likelihood + log_prior

  	@show log_likelihood, log_prior, log_posterior

	return log_posterior
end

function sample_gt(draw::Integer, gt::Matrix{Float64},
			shot_noise_img::Matrix{Float64})

   	if draw > initial_burn_in_period

# 		if draw % annealing_frequency <= annealing_frequency/2
  		temperature = 1.0 + (annealing_starting_temperature-1.0)*
     	    			exp(-((draw-1) % annealing_frequency)/annealing_time_constant)
#     	        	temperature = 
#     	        	(annealing_starting_temperature+1.0)-temperature		
#     	        	else
#     	           	       		temperature = 1.0 + (annealing_starting_temperature-1.0)*
#     	       				exp(-(((draw-1) % annealing_frequency)-
#     	       				      annealing_frequency/2)/annealing_time_constant)
#     	       		end
	else
   		temperature = 1.0
   	end
	@show temperature


	@everywhere workers() begin

		temperature = $temperature

		sub_gt::Matrix{Float64} = ($gt)[im_raw:ip_raw, jm_raw:jp_raw]
		sub_shot_noise_img::Matrix{Float64} =
						get_sub_image($shot_noise_img,
								im_raw, ip_raw, jm_raw, jp_raw)

		sub_gt, sub_shot_noise_img, n_accepted =
			sample_gt_neighborhood(temperature, sub_gt, sub_shot_noise_img)

		# Garbage Collection and free memory
 		GC.gc()

	end


 	local im::Integer
 	local ip::Integer
 	local jm::Integer
 	local jp::Integer
 	local sub_img::Matrix{Float64}
	local n_accept::Integer = 0


 	for i in 0:n_procs_per_dim_x-1
 		for j in 0:n_procs_per_dim_y-1

 			im = ghost_size + i*img_size_x/n_procs_per_dim_x + 1
 			ip = ghost_size + (i+1)*img_size_x/n_procs_per_dim_x
 			jm = ghost_size + j*img_size_y/n_procs_per_dim_y + 1
 			jp = ghost_size + (j+1)*img_size_y/n_procs_per_dim_y

 			sub_img = @fetchfrom (j*n_procs_per_dim_x+i+2) sub_gt
 			gt[im:ip, jm:jp] = sub_img[
 						ghost_size+1:end-ghost_size,
 						ghost_size+1:end-ghost_size]

 			sub_img = @fetchfrom (j*n_procs_per_dim_x+i+2) sub_shot_noise_img
 			shot_noise_img[im:ip, jm:jp] = sub_img[
                            ghost_size+1:end-ghost_size,
 							ghost_size+1:end-ghost_size]
			accepted = @fetchfrom (j*n_procs_per_dim_x+i+2) n_accepted
			n_accept += accepted
 		end
 	end

	println("accepted ", n_accept, " out of ", raw_image_size_x * raw_image_size_y, " pixels")	
	println("acceptance ratio = ", n_accept/ (raw_image_size_x * raw_image_size_y))	

	return gt, shot_noise_img
end

function save_data(current_draw::Integer,
					mcmc_log_posterior::Vector{Float64},
					gt::Matrix{Float64},
					shot_noise_image::Matrix{Float64},
					MAP_index::Integer,
					gt_MAP::Matrix{Float64},
					gt_mean::Matrix{Float64},
					averaging_counter::Float64)

	# Save the data in HDF5 format.
	file_name = string(working_directory, "mcmc_output_", current_draw, ".h5")

	fid = h5open(file_name,"w")
	write_dataset(fid, string("averaging_counter"), averaging_counter)
	write_dataset(fid, string("MAP_index"), MAP_index)
	write_dataset(fid, string("inferred_density"),
		      gt[ghost_size+1:end-ghost_size, ghost_size+1:end-ghost_size])
	write_dataset(fid, string("MAP_inferred_density"), 
		      gt_MAP[ghost_size+1:end-ghost_size, ghost_size+1:end-ghost_size])
	write_dataset(fid, string("mean_inferred_density"), 
		     gt_mean[ghost_size+1:end-ghost_size, ghost_size+1:end-ghost_size])
	write_dataset(fid, string("shot_noise_image"),
		shot_noise_image[ghost_size+1:end-ghost_size, ghost_size+1:end-ghost_size])
	write_dataset(fid, "mcmc_log_posterior",
			mcmc_log_posterior[1:current_draw])
	close(fid)

	return nothing
end
function plot_data(current_draw, gt, shot_noise_image, log_posterior)
	plot_gt  = heatmap(gt, legend=false, c=:grays)
	plot_shot  = heatmap(shot_noise_image, legend=false, c=:grays)
	plot_post  = plot(log_posterior[1:current_draw], legend=false, c=:grays)
	plot_raw  = heatmap(raw_image_with_ghosts, legend=false, c=:grays)


	l = @layout [a b; c d]
	display(plot(plot_raw, plot_post, plot_shot, plot_gt, layout = l, size=(2000, 2000) ))


	return nothing
end


function sampler_SIM(draws::Integer, initial_inferred_density::Matrix{Float64},
						initial_shot_noise_image::Matrix{Float64})

	# Initialize
	draw::Integer = 1
	println("draw = ", draw)
	flush(stdout);

  	gt::Matrix{Float64} = copy(initial_inferred_density)
	MAP_index::Integer = 1
   	MAP_gt::Matrix{Float64} = copy(gt)	
   	sum_gt::Matrix{Float64} =
 				zeros(img_size_x+2*ghost_size, img_size_y+2*ghost_size)
   	mean_gt::Matrix{Float64} =
 				zeros(img_size_x+2*ghost_size, img_size_y+2*ghost_size)

	shot_noise_image::Matrix{Float64} = 
                        copy(initial_shot_noise_image)

	mcmc_log_posterior::Vector{Float64} = zeros(draws)
  	mcmc_log_posterior[draw] =
  				compute_full_log_posterior(gt, shot_noise_image)

	averaging_counter::Float64 = 0.0
	plot_data(draw, gt, shot_noise_image, mcmc_log_posterior)

	for draw in 2:draws

		println("draw = ", draw)
		flush(stdout);

		gt, shot_noise_image =
					sample_gt(draw, gt, shot_noise_image)
  		mcmc_log_posterior[draw] =
  					compute_full_log_posterior(gt, shot_noise_image)

		if mcmc_log_posterior[draw] == maximum(mcmc_log_posterior[1:draw])
			MAP_index = draw
			MAP_gt = copy(gt)
		end

        	if (draw >= initial_burn_in_period) &&
       			(draw % annealing_frequency >= annealing_burn_in_period) &&
				(draw % averaging_frequency == 0)
 
 			averaging_counter += 1.0
 			sum_gt += copy(gt)
 			mean_gt = sum_gt ./ averaging_counter
 
 			println("Averaging Counter = ", averaging_counter)
 			println("Saving Data...")
 			flush(stdout);


         		save_data(draw, mcmc_log_posterior,
         				gt, shot_noise_image,
         				MAP_index,
         				MAP_gt,
         				mean_gt,
         				averaging_counter)
        	end
	
		if draw % plotting_frequency == 0 
 			plot_data(draw, mean_gt, shot_noise_image, mcmc_log_posterior)

		end
#        if (draw >= initial_burn_in_period) &&
#       			(draw % annealing_frequency == 0) 
#
#			gt[ghost_size+1:end-ghost_size,
# 				ghost_size+1:end-ghost_size] = rand(img_size_x, img_size_y) 
#		end


		# Garbage Collection and free memory
 		GC.gc()
	end

	return gt, shot_noise_image
end

file_name = string(working_directory,
				"poisson_image.tif")
grnd_truth_poisson = TiffImages.load(file_name)
grnd_truth_poisson = Float64.(grnd_truth_poisson)


inferred_density = zeros(img_size_x+2*ghost_size, img_size_y+2*ghost_size)
inferred_density[ghost_size+1:end-ghost_size,
 			ghost_size+1:end-ghost_size]=rand(img_size_x, img_size_y)

inferred_shot_noise_image = zeros(img_size_x+2*ghost_size, img_size_y+2*ghost_size)
inferred_shot_noise_image[ghost_size+1:end-ghost_size,
 		 ghost_size+1:end-ghost_size]=grnd_truth_poisson[:, :]

#inferred_shot_noise_image[
#                    ghost_size+1:end-ghost_size, 
#					ghost_size+1:end-ghost_size] =
#			round.(abs.((input_raw_image .- offset) ./ gain))

#heatmap(input_raw_image, c=:grays)
println("Starting sampler...")
flush(stdout);

inferred_density, inferred_shot_noise_image =
		sampler_SIM(total_draws, inferred_density, inferred_shot_noise_image)
rmprocs()
