function get_log_prior(object::Matrix{Float64})


 	val_range_x = collect(padding_size+1:1:padding_size+raw_img_size_x)
 	val_range_y = collect(padding_size+1:1:padding_size+raw_img_size_y)
 	mod_fft_image = vec(abs.(fftshift(fft(ifftshift(object))))[val_range_x, val_range_y]) .+ eps()
 	log_prior::Float64 = logpdf(Dirichlet((modulation_transfer_function_vectorized) .+ eps()), 
 									mod_fft_image ./ sum(mod_fft_image))

	return log_prior
end


function get_log_likelihood(object::Matrix{Float64},
			shot_noise_image::Matrix{Float64})

	log_likelihood::Float64 = 0.0

	mean_image::Matrix{Float64} = get_mean_image(object)
	val_range_x = collect(padding_size+1:1:padding_size+raw_img_size_x)
	val_range_y = collect(padding_size+1:1:padding_size+raw_img_size_y)

	log_likelihood += sum(logpdf.(Poisson.(
				mean_image[val_range_x, val_range_y] .+ eps()),
				shot_noise_image[val_range_x, val_range_y]))
	log_likelihood += sum(logpdf.(Normal.(
                (gain_map_with_padding[val_range_x, val_range_y] .*
				shot_noise_image[val_range_x, val_range_y]) .+
					offset_map_with_padding[val_range_x, val_range_y],
                    error_map_with_padding[val_range_x, val_range_y] .+ eps()),
					raw_image_with_padding[val_range_x, val_range_y]))

	return log_likelihood
end

function compute_full_log_posterior(object::Matrix{Float64},
							shot_noise_image::Matrix{Float64})

	log_likelihood::Float64 = get_log_likelihood(object, shot_noise_image)
 	log_prior::Float64 = get_log_prior(object)
	log_posterior::Float64 = log_likelihood + log_prior

 	@show log_likelihood, log_prior, log_posterior

	return log_posterior
end

