function light_factor = adjust_light_factor(light_factor)
for i = 1:numel(light_factor)
     if light_factor(i)>= 60 &&  light_factor(i)< 70
            light_factor(i)= 75;
        elseif light_factor(i)  >= 70 &&light_factor(i)< 80
            light_factor(i)  = 85;
        elseif light_factor(i) >= 80 && light_factor(i)< 90
            light_factor(i) = 90;
        elseif light_factor(i) >= 90
         if i < numel(light_factor)  
                continue; 
            else   
            return;
        end
    end
end