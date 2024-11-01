using TiffImages
using Statistics
using Plots

include("input_parameters.jl")
include("input_data.jl")

using Distributed

println("Adding processors...")
flush(stdout);
addprocs(n_procs_per_dim_x*n_procs_per_dim_y, topology=:master_worker)
println("Done.")
flush(stdout);

@everywhere using Random, Distributions, LinearAlgebra, SpecialFunctions, FFTW
@everywhere rng = MersenneTwister(myid());


const sub_raw_img_size_x::Int64 = raw_img_size_x/n_procs_per_dim_x 
const sub_raw_img_size_y::Int64 = raw_img_size_y/n_procs_per_dim_y

include("psf.jl")
include("image_formation.jl")
include("global_posterior.jl")
include("output.jl")

@everywhere workers() begin

	include("input_parameters.jl")
	include("chunk_images.jl")
	include("psf_for_inference.jl")
	include("image_formation_for_inference.jl")
	include("bayesian_inference.jl")

end

function sampler()

 	@show psf_type
 	@show abbe_diffraction_limit
 	@show physical_pixel_size
 	@show padding_size
 	@show median_photon_count

	# Initialize
	draw::Int64 = 1
	println("draw = ", draw)
	flush(stdout);

	# Arrays for main variables of interest
	object::Matrix{Float64} = zeros(raw_img_size_x+2*padding_size, raw_img_size_y+2*padding_size)
 	object[padding_size+1:end-padding_size,
 	       padding_size+1:end-padding_size] .= rand(rng, raw_img_size_x, raw_img_size_y)

	sum_object::Matrix{Float64} =
		zeros(raw_img_size_x+2*padding_size, raw_img_size_y+2*padding_size)
	mean_object::Matrix{Float64} =
	 	zeros(raw_img_size_x+2*padding_size, raw_img_size_y+2*padding_size)
	
	shot_noise_image::Matrix{Float64} = 
		zeros(raw_img_size_x+2*padding_size, raw_img_size_y+2*padding_size)
    	shot_noise_image[padding_size+1:end-padding_size, 
    		padding_size+1:end-padding_size] .= 
 			Int64.(ceil.(abs.((input_raw_image .- median(offset_map_with_padding)) ./ 
					  (median(gain_map_with_padding)))))

	intermediate_img::Matrix{Float64} = zeros(3*raw_img_size_x, 3*raw_img_size_y)
	apply_reflective_BC_object!(object, intermediate_img)
	apply_reflective_BC_shot!(shot_noise_image, intermediate_img)

	mcmc_log_posterior::Vector{Float64} = zeros(total_draws)
    	mcmc_log_posterior[draw] =
    				compute_full_log_posterior(object, shot_noise_image)

	averaging_counter::Float64 = 0.0
 
	@everywhere workers() begin
		n_accepted::Int64 = 0
		temperature::Float64 = 0.0

		sub_object::Matrix{Float64} = zeros(sub_raw_img_size_x+2*padding_size, sub_raw_img_size_y+2*padding_size)
 		sub_shot_noise_img::Matrix{Float64} = zeros(sub_raw_img_size_x+2*padding_size, sub_raw_img_size_y+2*padding_size)

		# Arrays to store intermediate variables
 		mean_img_ij = zeros(Float64, padding_size+1, padding_size+1)
		proposed_mean_img_ij = zeros(Float64, padding_size+1, padding_size+1)
		FFT_var = zeros(ComplexF64, 2*padding_size+1, 2*padding_size+1)
		iFFT_var = zeros(ComplexF64, 2*padding_size+1, 2*padding_size+1)
		img_ij = zeros(ComplexF64, 2*padding_size+1, 2*padding_size+1)
 		img_ij_abs = zeros(Float64, 2*padding_size+1, 2*padding_size+1)
 		mod_fft_img_ij = zeros(Float64, size(modulation_transfer_function_ij_vectorized)[1]) 
	end

 
	im::Vector{Int64} = zeros(n_procs_per_dim_x)
	ip::Vector{Int64} = zeros(n_procs_per_dim_x)
	jm::Vector{Int64} = zeros(n_procs_per_dim_y)
	jp::Vector{Int64} = zeros(n_procs_per_dim_y)

	for i in 0:n_procs_per_dim_x-1

		im[i+1] = padding_size + i*sub_raw_img_size_x + 1
		ip[i+1] = padding_size + (i+1)*sub_raw_img_size_x

	end
	for j in 0:n_procs_per_dim_y-1

		jm[j+1] = padding_size + j*sub_raw_img_size_y + 1
		jp[j+1] = padding_size + (j+1)*sub_raw_img_size_y

	end

	n_accept::Int64 = 0
	temperature::Float64 = 0.0

	for draw in 2:total_draws

		if draw > chain_burn_in_period
			temperature = 1.0 + (annealing_starting_temperature-1.0)*
					exp(-((draw-chain_burn_in_period-1) % 
					      annealing_frequency)/annealing_time_constant)
		elseif draw < chain_burn_in_period
       			temperature = 1.0 + (chain_starting_temperature-1.0)*
					exp(-((draw-1) % 
					      chain_burn_in_period)/chain_time_constant)
		end

		@show draw
		@show temperature
		flush(stdout);

		@everywhere workers() begin

  			temperature = $temperature
			draw = $draw
 
 			sub_object .= ($object)[im_raw:ip_raw, jm_raw:jp_raw]
  			sub_shot_noise_img .= ($shot_noise_image)[im_raw:ip_raw, jm_raw:jp_raw]

			n_accepted = 0

			if i_procs + j_procs < draw+1
				if draw > chain_burn_in_period
					sub_object, sub_shot_noise_img, n_accepted =
							sample_object_neighborhood!(temperature, 
											sub_object, 
											sub_shot_noise_img,
											mean_img_ij,
											proposed_mean_img_ij,
											FFT_var,
											iFFT_var,
											img_ij,
 											img_ij_abs,
											mod_fft_img_ij,
											n_accepted)
				else

					sub_object, sub_shot_noise_img, n_accepted =
							sample_object_neighborhood_MLE!(temperature, 
											sub_object, 
											sub_shot_noise_img,
											mean_img_ij,
											proposed_mean_img_ij,
											FFT_var,
											iFFT_var,
											img_ij,
 											img_ij_abs,
											mod_fft_img_ij,
											n_accepted)

				end
			end

		end

    	n_accept = 0
    	for i in 1:n_procs_per_dim_x
    		for j in 1:n_procs_per_dim_y

 				procs_id::Int64 = ((j-1)*n_procs_per_dim_x+(i-1)+2)
    
 				sub_img = view((@fetchfrom procs_id sub_object), :, :)
 				object[im[i]:ip[i], jm[j]:jp[j]] .= 
 						sub_img[padding_size+1:end-padding_size,
    							padding_size+1:end-padding_size]
 
    
				sub_img = view((@fetchfrom procs_id sub_shot_noise_img), :, :)	
 				shot_noise_image[im[i]:ip[i], jm[j]:jp[j]] .= 
 						sub_img[padding_size+1:end-padding_size,
    							padding_size+1:end-padding_size]
 
    			accepted::Int64 = @fetchfrom procs_id n_accepted
    			n_accept += accepted
    
    		end
    	end
		apply_reflective_BC_object!(object, intermediate_img)
		apply_reflective_BC_shot!(shot_noise_image, intermediate_img)
    
    	println("accepted ", n_accept, " out of ", raw_img_size_x * raw_img_size_y, " pixels")	
    	println("acceptance ratio = ", n_accept/ (raw_img_size_x * raw_img_size_y))	
    	flush(stdout);

    	mcmc_log_posterior[draw] =
    					compute_full_log_posterior(object, shot_noise_image)

		if (draw == chain_burn_in_period) || ((draw > chain_burn_in_period) &&
			((draw - chain_burn_in_period) % 
			 annealing_frequency > annealing_burn_in_period ||
			 (draw - chain_burn_in_period) % annealing_frequency == 0) &&
			((draw - chain_burn_in_period) % averaging_frequency == 0))

  			averaging_counter += 1.0
  			sum_object .+= object
  			mean_object .= sum_object ./ averaging_counter
  
  			println("Averaging Counter = ", averaging_counter)
  			println("Saving Data...")
  			flush(stdout);
 
 
       		save_data(draw, mcmc_log_posterior,
          				object, shot_noise_image,
          				mean_object,
          				averaging_counter)
		end

		if draw % plotting_frequency == 0 
       		plot_data(draw, object, mean_object, shot_noise_image, mcmc_log_posterior)
		end
	end

	return nothing
end

@time sampler()
rmprocs(workers())
