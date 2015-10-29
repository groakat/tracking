function [ output_args ] = saveDSSTResultsToFile( positions, video_path)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here

%     positions(frame,:) = [pos target_sz];
%     
%     time = time + toc;
%     
%     %ground_truth = [ground_truth(:,[2,1]) + (ground_truth(:,[4,3]) - 1) / 2 , ground_truth(:,[4,3])];
%     groundTruthForDrawings = [ground_truth(:,[2,1])-(ground_truth(:,[4,3]) - 1) / 2,ground_truth(:,[4,3])];
    
    %video_path = '/home/stratos/Documents/FutsalData/FirstHalf400frames/seq1/CAM1/black_10/imgs'

    if(video_path(end)~=filesep)
        video_path(end+1)=filesep;
    end

    splitPathStr = strsplit(video_path,filesep);
    tmpStr = cell2mat(splitPathStr(end-2));
    correctPath = [strjoin(splitPathStr(1:end-2),filesep), filesep];
    
    fileName = [tmpStr '_matlabDSST.txt'];
    finalFileName = [correctPath fileName];
    %groundTruthForDrawings = [ground_truth(:,[2,1])-(ground_truth(:,[4,3]) - 1) / 2,ground_truth(:,[4,3])];
            
    rect_positions = [positions(:,[2,1]) - (positions(:,[4,3])-1)/2, positions(:,[4,3])];
        
    %filename: name_matlab_dsst.txt  [x,y,w,h]
    csvwrite(finalFileName,rect_positions);
    disp(['wrote file to: ', finalFileName]);
    
end

