function temperature_factor = adjust_temperature_factor(temperature_factor)    
     for i = 1:numel(temperature_factor)
        if temperature_factor(i)  > 28 && temperature_factor(i)<= 30
            temperature_factor(i) = 27.5;
        elseif temperature_factor(i) >27 && temperature_factor(i)  <= 28
            temperature_factor(i) = 26.5;
        elseif temperature_factor(i) > 26 && temperature_factor(i) <= 27
            temperature_factor(i) = 25;
        elseif temperature_factor(i)> 24 && temperature_factor(i) <= 26
            if i < numel(temperature_factor)  
                continue; 
            else
                return; 
            end
        end
    end
end
