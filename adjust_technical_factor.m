function technical_factor = adjust_technical_factor(technical_factor)    
   for i = 1:numel(technical_factor)
       if technical_factor(i) > 80 && technical_factor(i)< 85
            technical_factor(i) = 87.5;
        elseif technical_factor(i) >= 85&& technical_factor(i)  < 90
            technical_factor(i) = 92.5;
        elseif technical_factor(i)  >= 90 && technical_factor(i) < 95
            technical_factor(i) = 95;
        elseif technical_factor(i)>= 95
            if i < numel(technical_factor)  
                continue; 
            else  
                return; 
            end
        end
    end
end