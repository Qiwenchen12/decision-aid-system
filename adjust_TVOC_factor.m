function TVOC_factor = adjust_TVOC_factor(TVOC_factor)    
     for i = 1:numel(TVOC_factor)
        if TVOC_factor(i)  >= 0.6 && TVOC_factor(i)< 0.8
            TVOC_factor(i) = 0.55;
        elseif TVOC_factor(i)  >= 0.5 && TVOC_factor(i) < 0.6
            TVOC_factor(i) = 0.4;
        elseif TVOC_factor(i) > 0.3 && TVOC_factor(i) < 0.5
            TVOC_factor(i) = 0.3;
        elseif TVOC_factor(i)<= 0.3
            if i < numel(TVOC_factor)  
                continue; 
            else
                return; 
            end
        end
    end
end
