function save_data(current_draw::Integer,
    mcmc_log_posterior::Vector{Float64},
    object::Matrix{Float64},
    shot_noise_image::Matrix{Float64},
    object_mean::Matrix{Float64},
    averaging_counter::Float64)


	grays = convert.(Gray{Float64}, 
		object[padding_size+1:padding_size+raw_img_size_x, 
			padding_size+1:padding_size+raw_img_size_y])
	img = TiffImages.DenseTaggedImage(grays)
	TiffImages.save(string(working_directory,
			"inferred_object_", averaging_counter,".tif"), img)
	
	grays = convert.(Gray{Float64}, 
		object_mean[padding_size+1:padding_size+raw_img_size_x, 
			padding_size+1:padding_size+raw_img_size_y])
	img = TiffImages.DenseTaggedImage(grays)
	TiffImages.save(string(working_directory,
			"mean_inferred_object_", averaging_counter,".tif"), img)
	
	return nothing
end

function plot_data(current_draw::Int64, object::Matrix{Float64}, mean_object::Matrix{Float64}, shot_noise_image::Matrix{Float64}, log_posterior::Vector{Float64})

	plot_object  = heatmap(view(object, 
			padding_size+1:padding_size+raw_img_size_x, 
			padding_size+1:padding_size+raw_img_size_y), 
			legend=false, 
			c=:grays, 
			yflip=true, 
			title = "Current Sample",
			titlefontsize = 20)

	plot_mean_object  = heatmap(view(mean_object, 
			padding_size+1:padding_size+raw_img_size_x, 
			padding_size+1:padding_size+raw_img_size_y), 
			legend=false, 
			c=:grays, 
			yflip=true, 
			title = "Mean",
			titlefontsize = 20)

	plot_shot  = heatmap(view(shot_noise_image, 
			padding_size+1:padding_size+raw_img_size_x, 
			padding_size+1:padding_size+raw_img_size_y), 
			legend=false, 
			c=:grays, 
			yflip=true, 
			title = "Shot Noise Image",
			titlefontsize = 20)

	plot_post  = plot(view(log_posterior,1:current_draw), 
			legend=false, 
			c=:grays, 
			title = "log(Posterior)",
			titlefontsize = 20)

	plot_raw  = heatmap(view(raw_image_with_padding, 
			padding_size+1:padding_size+raw_img_size_x, 
			padding_size+1:padding_size+raw_img_size_y), 
			legend=false, 
			c=:grays, 
			yflip=true, 
			title = "Raw Image",
			titlefontsize = 20)
	
	l = @layout [a b; c d]
	display(plot(plot_raw, 
				 plot_post, 
				 plot_object, 
				 plot_mean_object, 
				 layout = l, 
				 size=(2000, 2000) ))


return nothing
end
