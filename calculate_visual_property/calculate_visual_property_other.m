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

visual_matrix = cell(2,4);
visual_matrix{1,1} = 'rating';
visual_matrix{1,2} = rating_luminance;
visual_matrix{1,3} = rating_contrast;
visual_matrix{1,4} = rating_entropy;
visual_matrix{2,1} = 'fixation';
visual_matrix{2,2} = fixation_luminance;
visual_matrix{2,3} = fixation_contrast;
visual_matrix{2,4} = fixation_entropy;
writetable(cell2table(visual_matrix,...
    'VariableNames',{'name' 'luminance' 'contrast' 'complexity'}), 'RatingFixation_Visual_Property.csv')



 function E = get_entropy(I)
    % Assume I in the range 0..1
    p = hist(I(:), linspace(0,1,256));  % create histogram
    p(p==0) = []; % remove zero entries that would cause log2 to return NaN
    p = p/numel(I); % normalize histogram to unity
    E = -sum(p.*log2(p)); % entropy definition
return;

 end
 
 