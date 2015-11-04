function [positions, fps] = dsst(params,ground_truth)

    % [positions, fps] = dsst(params)
    
    
    status_object = dsst_initialize(params,ground_truth);
    
    
    
    for frame = 1:num_frames
        %load image
        %im = imread([video_path img_files{frame}]);
        %pause
        tic;
        
        if frame > 1
%             im = im(:,[20:end,1:19],:); %move image to the right
            
            newCurImSize = curImSize/1.0005;
            rectIm = [ceil(curImSize/2-newCurImSize/2), floor(curImSize/2+newCurImSize/2)];
            im = im(rectIm(1):rectIm(3),rectIm(2):rectIm(4),:);
            curImSize = newCurImSize;            
            im = mexResize(im, initImSize, 'auto');

            % extract the test sample feature map for the translation filter
            xt = get_translation_sample(im, pos, sz, currentScaleFactor, cos_window);

            % calculate the correlation response of the translation filter
            xtf = fft2(xt);
            response = real(ifft2(sum(hf_num .* xtf, 3) ./ (hf_den + lambda)));

            % find the maximum translation response
            [row, col] = find(response == max(response(:)), 1);

            % update the position
            pos = pos + round((-sz/2 + [row, col]) * currentScaleFactor);

            % extract the test sample feature map for the scale filter
            xs = get_scale_sample(im, pos, base_target_sz, currentScaleFactor * scaleFactors, scale_window, scale_model_sz);

            % calculate the correlation response of the scale filter
            xsf = fft(xs,[],2);
            scale_response = real(ifft(sum(sf_num .* xsf, 1) ./ (sf_den + lambda)));
            
            imScaleResponse = scale_response(floor(1:0.25:nScales+3*.025));
%             imScaleResponse = repmat(imScaleResponse',1,40);
            fig5H=figure(5);
%             imshow(imScaleResponse);
            plot(1:length(imScaleResponse), imScaleResponse, '-g');
            hold on
%             plot(1:40, (17*4-1.5)*ones(1,40),'g-','LineWidth',3)
            tmpLine = 0:0.05:2;%max(scale_response);
            plot((ceil(nScales/2.0)*4-1.5)*ones(1,length(tmpLine)), tmpLine,'b-');
            
            % find the maximum scale response
            recovered_scale = find(scale_response == max(scale_response(:)), 1);
            %plot((recovered_scale*4-1.5)*ones(1,length(tmpLine)), tmpLine,'r-');
            plot(recovered_scale*4-1.5, scale_response(recovered_scale),'r.','MarkerSize',10);
            hold off
            
            
            % update the scale
            currentScaleFactor = currentScaleFactor * scaleFactors(recovered_scale);
            if currentScaleFactor < min_scale_factor
                currentScaleFactor = min_scale_factor;
            elseif currentScaleFactor > max_scale_factor
                currentScaleFactor = max_scale_factor;
            end
        end

        % extract the training sample feature map for the translation filter
        xl = get_translation_sample(im, pos, sz, currentScaleFactor, cos_window);

        % calculate the translation filter update
        xlf = fft2(xl);
        new_hf_num = bsxfun(@times, yf, conj(xlf));  %yf is the fourier transform of the gaussian output
        new_hf_den = sum(xlf .* conj(xlf), 3);     %3 dimensional filter sum over the 3rd dimension

        % extract the training sample feature map for the scale filter
        xs = get_scale_sample(im, pos, base_target_sz, currentScaleFactor * scaleFactors, scale_window, scale_model_sz);

        % calculate the scale filter update
        xsf = fft(xs,[],2);
        new_sf_num = bsxfun(@times, ysf, conj(xsf));
        new_sf_den = sum(xsf .* conj(xsf), 1);


        if frame == 1
            % first frame, train with a single image
            hf_den = new_hf_den;
            hf_num = new_hf_num;

            sf_den = new_sf_den;
            sf_num = new_sf_num;
        else
            % subsequent frames, update the model
            hf_den = (1 - learning_rate) * hf_den + learning_rate * new_hf_den;
            hf_num = (1 - learning_rate) * hf_num + learning_rate * new_hf_num;
            sf_den = (1 - learning_rate) * sf_den + learning_rate * new_sf_den;
            sf_num = (1 - learning_rate) * sf_num + learning_rate * new_sf_num;
        end

        % calculate the new target size
        target_sz = floor(base_target_sz * currentScaleFactor);

        %save position
        positions(frame,:) = [pos target_sz];

        time = time + toc;

        %ground_truth = [ground_truth(:,[2,1]) + (ground_truth(:,[4,3]) - 1) / 2 , ground_truth(:,[4,3])];
        groundTruthForDrawings = [ground_truth(:,[2,1])-(ground_truth(:,[4,3]) - 1) / 2,ground_truth(:,[4,3])];

        %visualization
        if visualization == 1
            rect_position = [pos([2,1]) - target_sz([2,1])/2, target_sz([2,1])];
            if frame == 1,  %first frame, create GUI
                fig_handle=figure('Number','off', 'Name',['Tracker - ' video_path]);
                im_handle = imshow(uint8(im), 'Border','tight', 'InitialMag', 100 + 100 * (length(im) < 500));
                rect_handle = rectangle('Position',rect_position, 'EdgeColor','g');

                %ground truth drawing, for comparison only          
                rect_gt_handle = rectangle('Position',groundTruthForDrawings(frame,:), 'EdgeColor','r');

                text_handle = text(10, 10, ['current frame: ' int2str(frame)]);
                textScale_handle = text(10, 50, ['currentScale: ' num2str(currentScaleFactor)]);
                set(text_handle, 'color', [0 1 1]);
                set(textScale_handle, 'color', [0 1 1]);
                
                filterFig_handle=figure('Name', 'hf_num');
                hf_num_handles = cell(30);
                subplot(5,6,1);
                hf_num_handles{1} = imshow(uint8((xl(:,:,1) + 0.5) * 255), 'Border','tight');

                curFilterFourier = bsxfun(@times, conj(new_hf_num),1./new_hf_den);
                for i= 2:29
                    subplot(5,6,i);
                    tmp = real(ifft2(curFilterFourier(:,:,i-1)));
                    tmp=tmp/min(tmp(:));
                    set(hf_num_handles{i}, 'CData', fftshift(uint8( tmp* 255)))
                    hf_num_handles{i} = imshow(fftshift(uint8(tmp * 255)), 'Border','tight');
                end
                subplot(5,6,30);
                hf_num_handles{30} = imshow(uint8((xl(:,:,1) + 0.5) * 255), 'Border','tight');
                
                set(fig_handle,'OuterPosition',[1 1 500 400]);
                set(filterFig_handle,'OuterPosition',[1 501 800 600]);                
            else
                try  %subsequent frames, update GUI
                    set(im_handle, 'CData', im)


                    set(rect_handle, 'Position', rect_position)
                    %ground truth drawing, for comparison only
                    set(rect_gt_handle, 'Position', groundTruthForDrawings(frame,:))
                    set(text_handle, 'string', ['current frame: ' int2str(frame)]);
                    set(textScale_handle, 'string', ['currentScale: ' num2str(currentScaleFactor)]);
                    
                    set(hf_num_handles{1}, 'CData', uint8((xl(:,:,1) + 0.5) * 255))                
                    curFilterFourier = bsxfun(@times, conj(new_hf_num),1./new_hf_den);
                    for i= 2:29                    
                        tmp = real(ifft2(curFilterFourier(:,:,i-1)));
                        tmp=tmp/min(tmp(:));
                        set(hf_num_handles{i}, 'CData', fftshift(uint8( tmp* 255)))
                    end
                    set(hf_num_handles{30}, 'CData', uint8(response * 255))   
                catch
                    return
                end
            end
            
            figure(fig_handle)
            mycolormap = [1, 0, 0; 0, 1, 0];
            % assigen colormap
            colormap(mycolormap)
            hold on
            L = line(ones(2),ones(2), 'LineWidth',2);               % generate line 
            set(L,{'color'},mat2cell(mycolormap,ones(1,2),3));            % set the colors according to cmap
            legend('ground truth','tracker')  
            
            drawnow
    %         pause
        end
    end

    fps = num_frames/time;
end


function status_object = dsst_initialize(params,ground_truth)
    % parameters
    padding = params.padding;                         	%extra area surrounding the target
    output_sigma_factor = params.output_sigma_factor;	%spatial bandwidth (proportional to target)
    lambda = params.lambda;
    learning_rate = params.learning_rate;
    nScales = params.number_of_scales;
    scale_step = params.scale_step;
    scale_sigma_factor = params.scale_sigma_factor;
    scale_model_max_area = params.scale_model_max_area;

    video_path = params.video_path;
    img_files = params.img_files;
    pos = floor(params.init_pos);
    target_sz = floor(params.wsize);

    visualization = params.visualization;

    num_frames = numel(img_files);

    init_target_sz = target_sz;

    % target size att scale = 1
    base_target_sz = target_sz;

    % window size, taking padding into account
    sz = floor(base_target_sz * (1 + padding));

    % desired translation filter output (gaussian shaped), bandwidth
    % proportional to target size
    output_sigma = sqrt(prod(base_target_sz)) * output_sigma_factor;
    [rs, cs] = ndgrid((1:sz(1)) - floor(sz(1)/2), (1:sz(2)) - floor(sz(2)/2));
    y = exp(-0.5 * (((rs.^2 + cs.^2) / output_sigma^2)));
    yf = single(fft2(y));


    % desired scale filter output (gaussian shaped), bandwidth proportional to
    % number of scales
    scale_sigma = nScales/sqrt(33) * scale_sigma_factor;
    ss = (1:nScales) - ceil(nScales/2);
    ys = exp(-0.5 * (ss.^2) / scale_sigma^2);
    ysf = single(fft(ys));

    % store pre-computed translation filter cosine window
    cos_window = single(hann(sz(1)) * hann(sz(2))');

    % store pre-computed scale filter cosine window
    if mod(nScales,2) == 0
        scale_window = single(hann(nScales+1));
        scale_window = scale_window(2:end);         % probably does that to get the maximum factor of the hann window
    else
        scale_window = single(hann(nScales));
    end;
    %%%stratos change
    %scale_window=ones(1,nScales);%sqrt(sqrt(scale_window));

    % scale factors
    ss = 1:nScales;
    scaleFactors = scale_step.^(ceil(nScales/2) - ss);

    % compute the resize dimensions used for feature extraction in the scale
    % estimation
    scale_model_factor = 1;
    if prod(init_target_sz) > scale_model_max_area
        scale_model_factor = sqrt(scale_model_max_area/prod(init_target_sz));
    end
    scale_model_sz = floor(init_target_sz * scale_model_factor);

    currentScaleFactor = 1;

    % to calculate precision
    positions = zeros(numel(img_files), 4);

    % to calculate FPS
    time = 0;

    % find maximum and minimum scales
    im = imread([video_path img_files{1}]);
    min_scale_factor = scale_step ^ ceil(log(max(5 ./ sz)) / log(scale_step));   %5 is probably the minimum for height and width of the window
    max_scale_factor = scale_step ^ floor(log(min([size(im,1) size(im,2)] ./ base_target_sz)) / log(scale_step));

    
    %stratos test
    curImSize = size(im);
    curImSize = curImSize(1:2);
    initImSize = curImSize;
    
    fig4H = figure(4);
    set(fig4H,'Position',[1500 700 50 50]);
    fig5H = figure(5);
    set(fig5H,'OuterPosition',[800 500 700 200]);
    
    
    
    status_object.time = time;


end


function dsst_update()



end

