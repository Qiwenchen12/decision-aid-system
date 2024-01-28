function natural_factor = adjust_natural_factor(natural_factor)
for i = 1:numel(natural_factor)
     if natural_factor(i) >= 6 && natural_factor(i)< 8
            natural_factor(i) = 5;
        elseif natural_factor(i)>= 4 && natural_factor(i)  < 6
            natural_factor(i) = 3;
        elseif natural_factor(i) > 2 && natural_factor(i) < 4
            natural_factor(i) = 2;
        elseif natural_factor(i) <= 2
         if i < numel(natural_factor)  
                continue; 
            else   
            return;
        end
    end
end