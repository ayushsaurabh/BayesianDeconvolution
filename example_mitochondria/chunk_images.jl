const raw_img_size_x::Int64 = @fetchfrom 1 raw_img_size_x
const raw_img_size_y::Int64 = @fetchfrom 1 raw_img_size_y
const sub_raw_img_size_x::Int64 = @fetchfrom 1 sub_raw_img_size_x
const sub_raw_img_size_y::Int64 = @fetchfrom 1 sub_raw_img_size_y


# Cartesian coordinates for each processor
const i_procs::Int64 = (myid()-2) % n_procs_per_dim_x
const j_procs::Int64 = (myid()-2 - i_procs)/n_procs_per_dim_x

# Boundary coordinates of chunks or sub raw images in x an y direction
const im_raw::Int64 = i_procs*sub_raw_img_size_x + 1
const ip_raw::Int64 = 2*padding_size + (i_procs+1)*sub_raw_img_size_x
const jm_raw::Int64 = j_procs*sub_raw_img_size_y + 1
const jp_raw::Int64 = 2*padding_size + (j_procs+1)*sub_raw_img_size_y

const sub_img_size_x::Int64 = ip_raw - im_raw + 1 - 2*padding_size
const sub_img_size_y::Int64 = jp_raw - jm_raw + 1 - 2*padding_size




function get_sub_image(img::Matrix{Float64},
				im::Int64, ip::Int64, jm::Int64, jp::Int64)
	sub_img = img[im:ip, jm:jp]
	return sub_img
end

image = @fetchfrom 1 raw_image_with_padding
const sub_raw_image::Matrix{Float64} = get_sub_image(image, im_raw, ip_raw, jm_raw, jp_raw)

image = @fetchfrom 1 gain_map_with_padding
const sub_gain_map::Matrix{Float64} = get_sub_image(image, im_raw, ip_raw, jm_raw, jp_raw)

image = @fetchfrom 1 offset_map_with_padding
const sub_offset_map::Matrix{Float64} = get_sub_image(image, im_raw, ip_raw, jm_raw, jp_raw)

image = @fetchfrom 1 error_map_with_padding
const sub_error_map::Matrix{Float64} = get_sub_image(image, im_raw, ip_raw, jm_raw, jp_raw)

image = 0
GC.gc()
