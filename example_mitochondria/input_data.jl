function get_input_raw_image()
 	file_name = string(working_directory,
				"raw_image.tif")
   	img = TiffImages.load(file_name)
	img = reinterpret(UInt16, img)
   	img = Float64.(img)

    return img
end

const input_raw_image::Matrix{Float64} = get_input_raw_image()

const raw_img_size_x::Int64 = size(input_raw_image)[1]
const raw_img_size_y::Int64 = size(input_raw_image)[2]

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

        offset_map = offset .* ones(raw_img_size_x, raw_img_size_y)
        gain_map = gain .* ones(raw_img_size_x, raw_img_size_y)
        error_map = noise .* ones(raw_img_size_x, raw_img_size_y)

    end

    return offset_map, error_map, gain_map
end

const offset_map::Matrix{Float64}, error_map::Matrix{Float64}, gain_map::Matrix{Float64} = get_camera_calibration_data()

function add_padding_reflective_BC(input_img::Matrix{Float64})


	size_x::Int64 = size(input_img)[1]
	size_y::Int64 = size(input_img)[2]

	img::Matrix{Float64} = zeros(3*size_x, 3*size_y)
	img[size_x+1:end-size_x,
			size_y+1:end-size_y] .= input_img 
	img[1:size_x,
			size_y+1:end-size_y] .= input_img[end:-1:1, :] 
	img[end-size_x+1:end,
			size_y+1:end-size_y] .= input_img[end:-1:1, :] 
	img[size_x+1:end-size_x,
			1:size_y] .= input_img[:, end:-1:1] 
	img[size_x+1:end - size_x,
        end-size_y+1:end] .= input_img[:, end:-1:1] 
	img[1:size_x,1:size_y] .= input_img[end:-1:1, end:-1:1] 
	img[1:size_x,end-size_y+1:end] .= input_img[end:-1:1, end:-1:1] 
	img[end-size_x+1:end,1:size_y] .= input_img[end:-1:1, end:-1:1] 
	img[end-size_x+1:end,end-size_y+1:end] .= input_img[end:-1:1, end:-1:1] 

	return img[size_x-padding_size+1:2*size_x+padding_size, size_y-padding_size+1:2*size_y+padding_size]
end

const raw_image_with_padding::Matrix{Float64} = add_padding_reflective_BC(input_raw_image)
const gain_map_with_padding::Matrix{Float64} = add_padding_reflective_BC(gain_map)
const offset_map_with_padding::Matrix{Float64} = add_padding_reflective_BC(offset_map)
const error_map_with_padding::Matrix{Float64} = add_padding_reflective_BC(error_map)

function apply_reflective_BC_object!(object::Matrix{Float64}, intermediate_img::Matrix{Float64})

	size_x::Int64 = size(object)[1] - 2*padding_size
	size_y::Int64 = size(object)[2] - 2*padding_size

	intermediate_img[size_x+1:end-size_x,
			    size_y+1:end-size_y] .= object[padding_size+1:end-padding_size, padding_size+1:end-padding_size] 
	intermediate_img[1:size_x,
			    size_y+1:end-size_y] .= object[padding_size+1:end-padding_size, padding_size+1:end-padding_size][end:-1:1, :] 
	intermediate_img[end-size_x+1:end,
			    size_y+1:end-size_y] .= object[padding_size+1:end-padding_size, padding_size+1:end-padding_size][end:-1:1, :] 
	intermediate_img[size_x+1:end-size_x,
			    1:size_y] .= object[padding_size+1:end-padding_size, padding_size+1:end-padding_size][:, end:-1:1] 
	intermediate_img[size_x+1:end - size_x,
			    end-size_y+1:end] .= object[padding_size+1:end-padding_size, padding_size+1:end-padding_size][:, end:-1:1] 
	intermediate_img[1:size_x,1:size_y] .= object[padding_size+1:end-padding_size, padding_size+1:end-padding_size][end:-1:1, end:-1:1] 
	intermediate_img[1:size_x,end-size_y+1:end] .= object[padding_size+1:end-padding_size, padding_size+1:end-padding_size][end:-1:1, end:-1:1] 
	intermediate_img[end-size_x+1:end,1:size_y] .= object[padding_size+1:end-padding_size, padding_size+1:end-padding_size][end:-1:1, end:-1:1] 
	intermediate_img[end-size_x+1:end,end-size_y+1:end] .= object[padding_size+1:end-padding_size, padding_size+1:end-padding_size][end:-1:1, end:-1:1] 

	object .= intermediate_img[size_x-padding_size+1:2*size_x+padding_size, 
				   size_y-padding_size+1:2*size_y+padding_size]

	return nothing
end
function apply_reflective_BC_shot!(shot_noise_image::Matrix{Float64}, intermediate_img::Matrix{Float64})

	size_x::Int64 = size(shot_noise_image)[1] - 2*padding_size
	size_y::Int64 = size(shot_noise_image)[2] - 2*padding_size

	intermediate_img[size_x+1:end-size_x,
			    size_y+1:end-size_y] .= shot_noise_image[padding_size+1:end-padding_size, padding_size+1:end-padding_size] 
	intermediate_img[1:size_x,
			    size_y+1:end-size_y] .= shot_noise_image[padding_size+1:end-padding_size, padding_size+1:end-padding_size][end:-1:1, :] 
	intermediate_img[end-size_x+1:end,
			    size_y+1:end-size_y] .= shot_noise_image[padding_size+1:end-padding_size, padding_size+1:end-padding_size][end:-1:1, :] 
	intermediate_img[size_x+1:end-size_x,
			    1:size_y] .= shot_noise_image[padding_size+1:end-padding_size, padding_size+1:end-padding_size][:, end:-1:1] 
	intermediate_img[size_x+1:end - size_x,
			    end-size_y+1:end] .= shot_noise_image[padding_size+1:end-padding_size, padding_size+1:end-padding_size][:, end:-1:1] 
	intermediate_img[1:size_x,1:size_y] .= shot_noise_image[padding_size+1:end-padding_size, padding_size+1:end-padding_size][end:-1:1, end:-1:1] 
	intermediate_img[1:size_x,end-size_y+1:end] .= shot_noise_image[padding_size+1:end-padding_size, padding_size+1:end-padding_size][end:-1:1, end:-1:1] 
	intermediate_img[end-size_x+1:end,1:size_y] .= shot_noise_image[padding_size+1:end-padding_size, padding_size+1:end-padding_size][end:-1:1, end:-1:1] 
	intermediate_img[end-size_x+1:end,end-size_y+1:end] .= shot_noise_image[padding_size+1:end-padding_size, padding_size+1:end-padding_size][end:-1:1, end:-1:1] 

	shot_noise_image .= intermediate_img[size_x-padding_size+1:2*size_x+padding_size, 
				   size_y-padding_size+1:2*size_y+padding_size]

	return nothing
end

const median_photon_count = 
		median(abs.((input_raw_image .- median(offset_map_with_padding)) ./ (median(gain_map_with_padding))))

intermediate_img = 0
GC.gc()
