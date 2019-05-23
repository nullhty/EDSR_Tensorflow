clear; close all;
%% settings
dataDir = './DIV2K_train_HR/';  % 291 augment dataset
savepath = 'train_hdf5/train_x2.h5';  % save filename
size_input = 48; % 29 | 39
size_label = 96; % 57 | 77
up_scale = 2; % upsacling factor
stride_label = 96;
%% initialization


count_input = 0;
count_label = 0;
count_per = 0;
count_bic = 0;
per_size = 300000;
data = zeros(size_input, size_input, 3, 1);
label = zeros(size_label, size_label, 3, 1);

%% generate data
%% filepaths=dir(fullfile(folder,'*.bmp'));
f_lst = [];
f_lst = [f_lst; dir(fullfile(dataDir, '*.png'))];
file_num = 1;
for f_iter = 1:numel(f_lst)
    f_iter
	f_info = f_lst(f_iter);
    if f_info.name == '.'
        continue;
    end
    f_path = fullfile(dataDir,f_info.name);
    image0 = imread(f_path);
    if size(image0,3) == 1
        continue;
    end
    for angle = 0: 90 :0
		im_rot = imrotate(image0, angle);
		
		for scale = 1.0 : -0.1 : 1.0
			im_down = imresize(im_rot, scale, 'bicubic');
			
			for j = 3 : -2 : 3  % 3--> not flip, 1-->flip horizontally
                if j == 3
                    image = im_down;
                else
                    image = flip(im_down, j);
                end

				im_label = modcrop(image, up_scale); % high resolution subimage
				[hei_label,wid_label,~]=size(im_label);% HR subimage size
				
				for x = 1 : stride_label : hei_label -size_label+1
					for y = 1 : stride_label : wid_label -size_label+1
						
						target_patch=im_label(x:x+size_label-1,y:y+size_label-1,:);
						input_patch = imresize(target_patch, [size_input,size_input], 'bicubic');
						count_label=count_label+1;
						count_per = count_per + 1;
						label(:, :, :, count_per)=single(target_patch);
						data(:, :, :, count_per)=single(input_patch);
						%patch_name = sprintf('./EDSR_HR_train/%d', count_label);
						%save(patch_name, 'target_patch', 'input_patch', '-v6');
						if count_per >= per_size
							order = randperm(count_per);
							data = data(:, :, :, order);
							label = label(:, :, :, order);

							%data=permute(data,[4 3 1 2]);
							%label=permute(label,[4 3 1 2]);


							file_name = ['./train_data_div2k/train' num2str(file_num) '.h5'];
							h5create(file_name,'/data',size(data),'Datatype','single');
							h5create(file_name,'/label',size(label),'Datatype','single');

							h5write(file_name,'/data',data);
							h5write(file_name,'/label',label);
							file_num = file_num + 1;

							count_per = 0;
							data = [];
							label = [];
						end
					end
				end
			end
		end
    end
end

if count_per > 0
	order = randperm(count_per);
	data = data(:, :, :, order);
	label = label(:, :, :, order);

	%data=permute(data,[4 3 1 2]);
	%label=permute(label,[4 3 1 2]);

	file_name = ['./train_data_div2k/train' num2str(file_num) '.h5'];
	h5create(file_name,'/data',size(data),'Datatype','single');
	h5create(file_name,'/label',size(label),'Datatype','single');

	h5write(file_name,'/data',data);
	h5write(file_name,'/label',label);
	file_num = file_num + 1;

	count_per = 0;
	data = [];
	label = [];
end