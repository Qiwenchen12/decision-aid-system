function CO2_factor = adjust_CO2_factor(CO2_factor)    
   for i = 1:numel(CO2_factor)
        if CO2_factor(i) >= 0.15 &&  CO2_factor(i)< 0.25
            CO2_factor(i) = 0.125;
        elseif CO2_factor(i) >= 0.1 &&CO2_factor(i)  < 0.15
            CO2_factor(i) = 0.085;
        elseif CO2_factor(i)   > 0.07 &&  CO2_factor(i) < 0.1
            CO2_factor(i) = 0.07;
        elseif CO2_factor(i) <= 0.07
           if i < numel(CO2_factor)  
                continue; 
            else
                return; 
            end
        end
    end
end
