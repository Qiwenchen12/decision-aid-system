function humidity_factor = adjust_humidity_factor(humidity_factor)
for i = 1:numel(humidity_factor)
     if humidity_factor(i) >= 70 &&humidity_factor(i) < 80
            humidity_factor(i) = 67.5;
        elseif humidity_factor(i) >= 65 &&humidity_factor(i)  < 70
            humidity_factor(i) = 62.5;
        elseif humidity_factor(i) > 60 &&  humidity_factor(i)< 65
            humidity_factor(i)  = 60;
        elseif humidity_factor(i)<=60
         if i < numel(humidity_factor)  
                continue; 
            else   
            return;
        end
    end
end