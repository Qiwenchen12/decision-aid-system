function noise_factor = adjust_noise_factor(noise_factor)
for i = 1:numel(noise_factor)
     if noise_factor(i)>= 75 && noise_factor(i)< 85
            noise_factor(i)  = 72.5;
        elseif noise_factor(i)  >= 70 &&noise_factor(i)< 75
            noise_factor(i) = 65;
        elseif noise_factor(i) > 60 && noise_factor(i) < 70
            noise_factor(i) = 60;
        elseif noise_factor(i)<= 60
         if i < numel(noise_factor)  
                continue; 
            else   
            return;
        end
    end
end