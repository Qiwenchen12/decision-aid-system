function basicalconfiguration_factor = adjust_basicalconfiguration_factor(basicalconfiguration_factor)
     for i = 1:numel(basicalconfiguration_factor)
        if basicalconfiguration_factor(i) > 80 && basicalconfiguration_factor(i)< 85
            basicalconfiguration_factor(i) = 87.5;
        elseif basicalconfiguration_factor(i) >= 85&& basicalconfiguration_factor(i)  < 90
            basicalconfiguration_factor(i) = 92.5;
        elseif basicalconfiguration_factor(i)  >= 90 && basicalconfiguration_factor(i) < 95
            basicalconfiguration_factor(i) = 95;
        elseif basicalconfiguration_factor(i)>= 95
            if i < numel(basicalconfiguration_factor)  
                continue; 
            else
            return;
        end
    end
end
