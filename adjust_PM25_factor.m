function PM25_factor = adjust_PM25_factor(PM25_factor)
for i = 1:numel(PM25_factor)
     if PM25_factor(i)  >= 115 &&PM25_factor(i)< 150
            PM25_factor(i) = 95;
        elseif PM25_factor(i)  >= 75 && PM25_factor(i)< 115
            PM25_factor(i) = 55;
        elseif PM25_factor(i)  > 35 && PM25_factor(i) < 75
            PM25_factor(i)  = 35;
        elseif PM25_factor(i) <= 35
         if i < numel(PM25_factor)  
                continue; 
            else   
            return;
        end
    end
end
