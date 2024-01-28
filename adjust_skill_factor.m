function skill_factor = adjust_skill_factor(skill_factor)  
for i = 1:numel(skill_factor)
     if skill_factor(i)> 60 && skill_factor(i)< 70
            skill_factor(i)  = 75;
        elseif skill_factor(i) >= 70 && skill_factor(i)< 80
            skill_factor(i)  = 85;
        elseif skill_factor(i)>= 80 &&  skill_factor(i)< 90
            skill_factor(i)= 90;
        elseif skill_factor(i)>= 90
         if i < numel(skill_factor)  
                continue; 
            else   
            return;
        end
    end
end
