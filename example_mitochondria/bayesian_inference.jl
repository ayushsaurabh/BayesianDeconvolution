function log_Beta(a::Vector{Float64})
	z::Float64 = sum(loggamma.(a)) - loggamma(sum(a))
	return z
end

const log_Beta_MTF::Float64= log_Beta(modulation_transfer_function_ij_vectorized .+ eps())
const MTF_minus_one::Vector{Float64} = (conc_parameter .* modulation_transfer_function_ij_vectorized) .- 1.0


function log_Dirichlet(x::Vector{Float64}, a::Vector{Float64})
	z::Float64 = sum( (a .- 1.0)  .* log.(x)) - log_Beta(a)
	return z
end
function log_Dirichlet_MTF(x::Vector{Float64})
	x .= MTF_minus_one .* log.(x)
	z::Float64 = sum(x) - log_Beta_MTF
	return z
end

function get_log_likelihood_ij(i::Int64, j::Int64,
				mean_img_ij::Matrix{Float64},
				shot_noise_img_ij)

	i_minus::Int64 = 1
	i_plus::Int64 = padding_size+1


	j_minus::Int64 = 1
	j_plus::Int64 = padding_size+1

 	log_likelihood::Float64 = 0.0 
 	for jj in j_minus:j_plus
 		for ii in i_minus:i_plus
			log_likelihood += logpdf(Poisson(mean_img_ij[ii, jj] + eps()), 
						 	shot_noise_img_ij[ii, jj])
 		end
 	end


	return log_likelihood 
end


function get_log_prior_ij!(FFT_var::Matrix{ComplexF64},
			img_ij::Matrix{ComplexF64},
			img_ij_abs::Matrix{Float64},
			mod_fft_img_ij::Vector{Float64},
			i::Int64, j::Int64)

   	fftshift!(img_ij, FFT_var)
	img_ij_abs .= abs.(img_ij) 
	mod_fft_img_ij .= (vec(img_ij_abs) ./ sum(img_ij_abs)) .+ eps()

 	return log_Dirichlet_MTF(mod_fft_img_ij)
end

# The following Gibbs sampler computes likelihood for the neighborhood only.
function sample_object_neighborhood!(temperature::Float64,
				object::Matrix{Float64},
				shot_noise_image::Matrix{Float64},
				mean_img_ij::Matrix{Float64},
				proposed_mean_img_ij::Matrix{Float64},
				FFT_var::Matrix{ComplexF64},
				iFFT_var::Matrix{ComplexF64},
				img_ij::Matrix{ComplexF64},
				img_ij_abs::Matrix{Float64},
				mod_fft_img_ij::Vector{Float64},
				n_accepted::Int64)

	for j in padding_size+1:padding_size+sub_img_size_y
		for i in padding_size+1:padding_size+sub_img_size_x

			old_value::Float64 = object[i, j]

   			shot_noise_img_ij = view(shot_noise_image, 
    					i - half_padding_size:i + half_padding_size, 
    					j - half_padding_size:j + half_padding_size)


  			obj_ij = view(object, 
  				i - padding_size:i + padding_size, 
  				j - padding_size:j + padding_size)
     			ifftshift!(img_ij, obj_ij)
  			mul!(FFT_var, fft_plan, img_ij)

			old_log_prior::Float64 = 
     				get_log_prior_ij!(FFT_var, 
   						img_ij, 
    					img_ij_abs, 
    					mod_fft_img_ij, 
    					i, j)  

 			# FFT_var gets modified in the following function
    		mean_img_ij .= get_widefield_image_ij!(mean_img_ij, 
    					FFT_var,
    					iFFT_var,
    					img_ij,
      					i, j)

 			old_log_likelihood::Float64 = get_log_likelihood_ij(i, j,
 							mean_img_ij, shot_noise_img_ij)
 
 			old_log_posterior::Float64 = old_log_likelihood + old_log_prior
 			old_jac::Float64 = log(old_value)

 			proposed_value::Float64 = rand(rng, Normal(log(old_value), 
							covariance_object), 1)[1]
			proposed_value = exp.(proposed_value)
 			object[i, j] = proposed_value

			proposed_obj_ij = view(object, 
    				i - padding_size:i + padding_size, 
    				j - padding_size:j + padding_size)
       			ifftshift!(img_ij, proposed_obj_ij)
    			mul!(FFT_var, fft_plan, img_ij)

			proposed_log_prior::Float64 = 
       				get_log_prior_ij!(FFT_var, 
     						img_ij, 
      						img_ij_abs, 
      						mod_fft_img_ij, 
      						i, j)  

    		proposed_mean_img_ij .= get_widefield_image_ij!(proposed_mean_img_ij, 
    					FFT_var,
    					iFFT_var,
    					img_ij,
      					i, j)

  			proposed_log_likelihood::Float64 = get_log_likelihood_ij(i, j,
  						proposed_mean_img_ij, shot_noise_img_ij)
  
   			proposed_jac::Float64 = log(proposed_value)
   			proposed_log_posterior::Float64 = proposed_log_likelihood + proposed_log_prior
  
  			log_hastings::Float64 = (1.0/temperature) *
                          	(proposed_log_posterior - old_log_posterior) +
  							proposed_jac - old_jac
			log_rand::Float64 = log(rand(rng, Float64))
  
   			if log_hastings > log_rand
 				copy!(mean_img_ij, proposed_mean_img_ij)
   				n_accepted += 1
			else
				object[i, j] = old_value
			end

 
			# Sample Intermediate Expected Photon Counts on each Pixel
			# Choose the central pixel in the mean image
			# for expected photon count

  			expected_photon_count::Float64 = mean_img_ij[
  							half_padding_size+1, 
  							half_padding_size+1]
  
    		old_log_likelihood = logpdf(Normal(sub_gain_map[i, j]*
 					shot_noise_image[i, j] +
 					sub_offset_map[i, j],
 					sub_error_map[i, j] .+ eps()),
   					sub_raw_image[i, j])
  
 			old_log_prior = logpdf(Poisson(expected_photon_count + eps()),
    								shot_noise_image[i, j])
  
 			old_log_posterior = old_log_likelihood  + old_log_prior
   
    			proposed_shot_noise_pixel::Float64 =
 				rand(rng, Poisson(shot_noise_image[i, j] + eps()), 1)[1]
  
			new_log_likelihood = logpdf(Normal(sub_gain_map[i, j]*
            			proposed_shot_noise_pixel +
    					sub_offset_map[i, j], 
   					sub_error_map[i, j] .+ eps()),
   					sub_raw_image[i, j])
  
 			new_log_prior = logpdf(Poisson(expected_photon_count + eps()),
    						proposed_shot_noise_pixel)
  
 			new_log_posterior = new_log_likelihood  + new_log_prior
  
  			log_forward_proposal_probability::Float64 = 
 				logpdf(Poisson(shot_noise_image[i, j] + eps()),
    									proposed_shot_noise_pixel)
  
  			log_backward_proposal_probability::Float64 = 
 					logpdf(Poisson(proposed_shot_noise_pixel + eps()),
    									shot_noise_image[i, j])
   
			log_hastings = (1.0/temperature)*
            			(new_log_posterior - old_log_posterior) +
  						log_backward_proposal_probability -
  						log_forward_proposal_probability
   
 			log_rand = log(rand(rng, Float64))
    		if log_hastings > log_rand
    			shot_noise_image[i, j] = proposed_shot_noise_pixel
    		end

		end
	end

	return object, shot_noise_image, n_accepted
end

function sample_object_neighborhood_MLE!(temperature::Float64,
				object::Matrix{Float64},
				shot_noise_image::Matrix{Float64},
				mean_img_ij::Matrix{Float64},
				proposed_mean_img_ij::Matrix{Float64},
				FFT_var::Matrix{ComplexF64},
				iFFT_var::Matrix{ComplexF64},
				img_ij::Matrix{ComplexF64},
				img_ij_abs::Matrix{Float64},
				mod_fft_img_ij::Vector{Float64},
				n_accepted::Int64)

	for j in padding_size+1:padding_size+sub_img_size_y
		for i in padding_size+1:padding_size+sub_img_size_x

			old_value::Float64 = object[i, j]

   			shot_noise_img_ij = view(shot_noise_image, 
    					i - half_padding_size:i + half_padding_size, 
    					j - half_padding_size:j + half_padding_size)


  			obj_ij = view(object, 
  				i - padding_size:i + padding_size, 
  				j - padding_size:j + padding_size)
     			ifftshift!(img_ij, obj_ij)
  			mul!(FFT_var, fft_plan, img_ij)

 			# FFT_var gets modified in the following function
    		mean_img_ij .= get_widefield_image_ij!(mean_img_ij, 
    					FFT_var,
    					iFFT_var,
    					img_ij,
      					i, j)

 			old_log_likelihood::Float64 = get_log_likelihood_ij(i, j,
 							mean_img_ij, shot_noise_img_ij)
 
 			old_log_posterior::Float64 = old_log_likelihood
 			old_jac::Float64 = log(old_value)

 			proposed_value::Float64 = rand(rng, Normal(log(old_value), 
							covariance_object), 1)[1]
			proposed_value = exp.(proposed_value)
 			object[i, j] = proposed_value

			proposed_obj_ij = view(object, 
    				i - padding_size:i + padding_size, 
    				j - padding_size:j + padding_size)
       		ifftshift!(img_ij, proposed_obj_ij)
    		mul!(FFT_var, fft_plan, img_ij)

    		proposed_mean_img_ij .= get_widefield_image_ij!(proposed_mean_img_ij, 
    					FFT_var,
    					iFFT_var,
    					img_ij,
      					i, j)

  			proposed_log_likelihood::Float64 = get_log_likelihood_ij(i, j,
  						proposed_mean_img_ij, shot_noise_img_ij)
  
   			proposed_jac::Float64 = log(proposed_value)
   			proposed_log_posterior::Float64 = proposed_log_likelihood 
  
  			log_hastings::Float64 = (1.0/temperature) *
                          	(proposed_log_posterior - old_log_posterior) +
  							proposed_jac - old_jac
  
   			if log_hastings > 0.0
 				copy!(mean_img_ij, proposed_mean_img_ij)
   				n_accepted += 1
			else
				object[i, j] = old_value
			end

 
			# Sample Intermediate Expected Photon Counts on each Pixel
			# Choose the central pixel in the mean image
			# for expected photon count

  			expected_photon_count::Float64 = mean_img_ij[
  							half_padding_size+1, 
  							half_padding_size+1]
  
    		old_log_likelihood = logpdf(Normal(sub_gain_map[i, j]*
 					shot_noise_image[i, j] +
 					sub_offset_map[i, j],
 					sub_error_map[i, j] .+ eps()),
   					sub_raw_image[i, j])
  
 			old_log_prior = logpdf(Poisson(expected_photon_count + eps()),
    								shot_noise_image[i, j])
  
 			old_log_posterior = old_log_likelihood  + old_log_prior
   
    			proposed_shot_noise_pixel::Float64 =
 				rand(rng, Poisson(shot_noise_image[i, j] + eps()), 1)[1]
  
			new_log_likelihood = logpdf(Normal(sub_gain_map[i, j]*
            			proposed_shot_noise_pixel +
    					sub_offset_map[i, j], 
   					sub_error_map[i, j] .+ eps()),
   					sub_raw_image[i, j])
  
 			new_log_prior = logpdf(Poisson(expected_photon_count + eps()),
    						proposed_shot_noise_pixel)
  
 			new_log_posterior = new_log_likelihood  + new_log_prior
  
  			log_forward_proposal_probability::Float64 = 
 				logpdf(Poisson(shot_noise_image[i, j] + eps()),
    									proposed_shot_noise_pixel)
  
  			log_backward_proposal_probability::Float64 = 
 					logpdf(Poisson(proposed_shot_noise_pixel + eps()),
    									shot_noise_image[i, j])
   
			log_hastings = (1.0/temperature)*
            			(new_log_posterior - old_log_posterior) +
  						log_backward_proposal_probability -
  						log_forward_proposal_probability

  
    		if log_hastings > 0.0
    			shot_noise_image[i, j] = proposed_shot_noise_pixel
    		end

		end
	end

	return object, shot_noise_image, n_accepted
end

