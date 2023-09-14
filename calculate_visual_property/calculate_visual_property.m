clear all


video_folder = '../AffVids_Videos/';

f = [dir(fullfile(video_folder,'*mov')); dir(fullfile(video_folder,'*m4v'))];


all_video_matrix = [];
for k = 1:length(f)
  baseFileName = f(k).name;
  fullFileName = fullfile(video_folder, baseFileName);
  fprintf(1, 'Now reading %s\n', fullFileName);
  vid=VideoReader(fullFileName);
  
  n = vid.NumberOfFrames;
  duration = round(vid.Duration);
  
  % temporary save video information
  video_matrix = cell(duration,5);
  video_matrix(:,1) = repmat({baseFileName}, duration,1);
  video_matrix(:,2) =repmat({extractBetween(baseFileName,"_",".")},duration,1);
  % counter for each frame
  counter =1;
  
 % cut the videos into frames
 for i = 1:round(n/duration+1):n
    frames = read(vid,i);
    % imwrite(frames,['Frame' int2str(i), '.jpg']);
    
    % calculate luminance
    % 1. without tool box:
    % R=frames(:,:,1);
    % G=frames(:,:,2);
    % B=frames(:,:,3);
    % luminance = 0.299 * R + 0.587 * G + 0.114 * B;
    % mean_luminance = mean(luminance,'all');

    % 2. luminance with toolbox:
    mean_luminance = mean2(rgb2gray(frames))
    
    % calclate contrast
    mean_contrast = max(luminance(:)) - min(luminance(:));
    
    % calculate entropy/complexity of 
    frame_entropy = get_entropy(frames);
    
    % save
    video_matrix{counter,3} = mean_luminance;
    video_matrix{counter,4} = mean_contrast;
    video_matrix{counter,5} = frame_entropy;
    
    counter = counter + 1;
 end
 
% concatenate all videos
all_video_matrix = vertcat(all_video_matrix, video_matrix);

end

% save
writetable(cell2table(all_video_matrix,...
    'VariableNames',{'video_name' 'video_number' 'luminance' 'contrast' 'complexity'}), 'all_Visual_Property.csv')


% calculate rating period
rating = imread('Rating.png');

R=rating(:,:,1);
G=rating(:,:,2);
B=rating(:,:,3);
luminance = 0.299 * R + 0.587 * G + 0.114 * B;
rating_luminance = mean(luminance,'all');

% calclate contrast
rating_contrast = max(luminance(:)) - min(luminance(:));

% calculate entropy/complexity of 
rating_entropy = get_entropy(rating);


fixation = imread('Fixation.png');
% calculate fixation period
R=fixation(:,:,1);
G=fixation(:,:,2);
B=fixation(:,:,3);
luminance = 0.299 * R + 0.587 * G + 0.114 * B;
fixation_luminance = mean(luminance,'all');

% calclate contrast
fixation_contrast = max(luminance(:)) - min(luminance(:));

% calculate entropy/complexity of 
fixation_entropy = get_entropy(fixation);



 function E = get_entropy(I)
    % Assume I in the range 0..1
    p = hist(I(:), linspace(0,1,256));  % create histogram
    p(p==0) = []; % remove zero entries that would cause log2 to return NaN
    p = p/numel(I); % normalize histogram to unity
    E = -sum(p.*log2(p)); % entropy definition
return;

 end