function CO_factor = adjust_CO_factor(CO_factor)
   for i = 1:numel(CO_factor)
        if CO_factor(i) >= 10 &&  CO_factor(i)< 15
           CO_factor(i) = 8;
        elseif CO_factor(i) >= 6 &&  CO_factor(i)  < 10
            CO_factor(i) = 4.5;
        elseif CO_factor(i)  > 3 &&  CO_factor(i)< 6
            CO_factor(i)  = 3;
        elseif CO_factor(i)<= 3
         if i < numel(CO_factor)  
                continue; 
            else
            return;
        end
    end
end
