function bacteria_factor = adjust_bacteria_factor(bacteria_factor)
for i = 1:numel(bacteria_factor)
   if bacteria_factor(i) >= 4000 && bacteria_factor(i) < 7000
            bacteria_factor(i) = 3250;
        elseif bacteria_factor(i) >= 2500&& bacteria_factor(i) < 4000
            bacteria_factor(i) = 1600;
        elseif bacteria_factor(i)  > 700 && bacteria_factor(i) < 2500
            bacteria_factor(i) = 700;
        elseif bacteria_factor(i) <= 700
         if i < numel(bacteria_factor)  
                continue; 
            else
                return;
        end
    end
end




