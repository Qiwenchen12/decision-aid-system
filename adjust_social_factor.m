function social_factor = adjust_social_factor(social_factor)
for i = 1:numel(social_factor)
    if social_factor(i) == 0.4 
            social_factor(i) = 0.6;
        elseif social_factor(i) == 0.6
            social_factor(i) = 0.8;
        elseif social_factor(i) == 0.8
            social_factor(i) = 1.0;
        elseif social_factor(i) == 1.0
         if i < numel(social_factor)  
                continue; 
            else    
            return;
        end
    end
end
