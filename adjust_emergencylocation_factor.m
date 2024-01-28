function emergencylocation_factor = adjust_emergencylocation_factor(emergencylocation_factor)
   for i = 1:numel(emergencylocation_factor)
        if emergencylocation_factor(i) > 60 && emergencylocation_factor(i)< 70
            emergencylocation_factor(i) = 75;
        elseif emergencylocation_factor(i) >= 70 && emergencylocation_factor(i)  < 80
            emergencylocation_factor(i) = 85;
        elseif emergencylocation_factor(i)>= 80 && emergencylocation_factor(i) < 90
            emergencylocation_factor(i) = 90;
        elseif emergencylocation_factor(i)>= 90
         if i < numel(emergencylocation_factor)  
                continue; 
            else
            return;
        end
    end
end