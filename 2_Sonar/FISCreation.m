function fis=FISCreation(data,radius)
    % 如果没有指定 radius，使用默认值
    if ~exist('radius','var')
        radius = 0.3;  % 默认半径，可根据数据调整
    end
    
    x = data.TrainInputs;
    t = data.TrainTargets;
    
    % 使用 genfis2 生成 Sugeno 型 FIS
    fis = genfis2(x, t, radius);
end