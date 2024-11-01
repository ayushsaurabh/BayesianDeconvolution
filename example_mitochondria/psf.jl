grid_physical_1D_x = dx .* collect(-(raw_img_size_x/2 + padding_size):
			(raw_img_size_x/2 + padding_size - 1)) # in micrometers
grid_physical_1D_y = dx .* collect(-(raw_img_size_y/2 + padding_size):
			(raw_img_size_y/2 + padding_size - 1)) # in micrometers

#grid_physical_length_x = grid_physical_1D_x[end] - grid_physical_1D_x[1]			
grid_physical_length_x = (raw_img_size_x + 2*padding_size - 1)*dx			
grid_physical_length_y = (raw_img_size_y + 2*padding_size - 1)*dx			


df_x = 1/(grid_physical_length_x) # Physcially correct spacing in spatial frequency space in units of micrometer^-1 
df_y = 1/(grid_physical_length_y) # Physcially correct spacing in spatial frequency space in units of micrometer^-1 

f_corrected_grid_1D_x = df_x .* collect(-(raw_img_size_x/2 + padding_size):
			(raw_img_size_x/2 + padding_size - 1)) # in units of micrometer^-1
f_corrected_grid_1D_y = df_y .* collect(-(raw_img_size_y/2 + padding_size):
			(raw_img_size_y/2 + padding_size - 1)) # in units of micrometer^-1

mtf_on_grid = zeros(raw_img_size_x+2*padding_size, raw_img_size_y+2*padding_size)
psf_on_grid = zeros(raw_img_size_x+2*padding_size, raw_img_size_y+2*padding_size)

if psf_type == "airy_disk"
 	function incoherent_PSF(x_c::Vector{Float64}, x_e::Vector{Float64})
 		f_number::Float64 = 1/(2*numerical_aperture) ##approx
 		return (jinc(norm(x_c - x_e)/(light_wavelength*f_number)))^2
 	end
 
 	for j in 1:raw_img_size_y + 2*padding_size
 		for i in 1:raw_img_size_x + 2*padding_size
 			x_e::Vector{Float64} = [grid_physical_1D_x[i], grid_physical_1D_y[j]]
 			psf_on_grid[i, j] =  incoherent_PSF([0.0, 0.0], x_e)
 		end
 	end
 	normalization = sum(psf_on_grid) * dx^2
 	psf_on_grid = psf_on_grid ./ normalization
	intermediate_img = fftshift(fft(ifftshift(psf_on_grid)))

	function MTF(f_vector::Vector{Float64})
		f_number::Float64 = 1/(2*numerical_aperture) ##approx
		highest_transmitted_frequency = 1.0 / (light_wavelength*f_number) 
		norm_f = norm(f_vector) / highest_transmitted_frequency
		if norm_f < 1.0
			mtf = 2.0/pi * (acos(norm_f) - norm_f*sqrt(1 - norm_f^2))
		elseif norm_f > 1.0
			mtf = 0.0
		end
		return mtf
	end

	for j in 1:raw_img_size_y + 2*padding_size
		for i in 1:raw_img_size_x + 2*padding_size
				mtf_on_grid[i, j] = MTF([f_corrected_grid_1D_x[i], f_corrected_grid_1D_y[j]]) 
				if mtf_on_grid[i, j] == 0.0
					intermediate_img[i, j] = 0.0 * intermediate_img[i, j]
				end
		end
	end
	const FFT_point_spread_function::Matrix{ComplexF64} = ifftshift(intermediate_img) 

elseif psf_type == "gaussian"
	function incoherent_PSF(x_c::Vector{Float64}, x_e::Vector{Float64})
		return exp(-norm(x_c-x_e)^2/(2.0*sigma^2)) /
					(sqrt(2.0*pi) * sigma)^(size(x_e)[1])
	end
	for j in 1:raw_img_size_y + 2*padding_size
		for i in 1:raw_img_size_x + 2*padding_size
			x_e::Vector{Float64} = [grid_physical_1D_x[i], grid_physical_1D_y[j]]
			psf_on_grid[i, j] =  incoherent_PSF([0.0, 0.0], x_e)
		end
	end
	const FFT_point_spread_function::Matrix{ComplexF64} = fft(ifftshift(psf_on_grid))
end



const modulation_transfer_function::Matrix{Float64} = abs.(fftshift(FFT_point_spread_function))[
 				padding_size+1:end-padding_size, padding_size+1:end-padding_size] 
const modulation_transfer_function_vectorized::Vector{Float64} = 
			vec(modulation_transfer_function) ./ sum(modulation_transfer_function)

psf_on_grid = 0
mtf_on_grid = 0
grid_physical_1D_x = 0
grid_physical_1D_y = 0
f_corrected_grid_1D_x = 0
f_corrected_grid_1D_y = 0
intermediate_img = 0

GC.gc()
