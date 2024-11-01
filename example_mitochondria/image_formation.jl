function get_widefield_image(object::Matrix{Float64})
	FFT_illuminated_object::Matrix{ComplexF64} =
				fft(ifftshift(object)) .* dx^2
	FFT_final::Matrix{ComplexF64} =
				FFT_point_spread_function .* FFT_illuminated_object

 	image::Matrix{Float64} = abs.(real.(fftshift(ifft(FFT_final))))

	return image
end

function get_mean_image(object::Matrix{Float64})

	final_image::Matrix{Float64} =
		get_widefield_image(object)

	return final_image
end


