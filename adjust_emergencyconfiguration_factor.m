function emergencyconfiguration_factor = adjust_emergencyconfiguration_factor(emergencyconfiguration_factor)    
     for i = 1:numel(emergencyconfiguration_factor)
        if emergencyconfiguration_factor(i) > 80 && emergencyconfiguration_factor(i)< 85
            emergencyconfiguration_factor(i) = 87.5;
        elseif emergencyconfiguration_factor(i) >= 85&& emergencyconfiguration_factor(i)  < 90
            emergencyconfiguration_factor(i) = 92.5;
        elseif emergencyconfiguration_factor(i)  >= 90 && emergencyconfiguration_factor(i) < 95
            emergencyconfiguration_factor(i) = 95;
        elseif emergencyconfiguration_factor(i)>= 95
            if i < numel(emergencyconfiguration_factor)  
                continue; 
            else   
                return; 
            end
        end
    end
end
