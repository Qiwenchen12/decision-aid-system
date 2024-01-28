function plan_factor = adjust_plan_factor(plan_factor)
for i = 1:numel(plan_factor)
     if plan_factor(i)  > 60 && plan_factor(i)< 70
            plan_factor(i) = 75;
        elseif plan_factor(i)  >= 70 && plan_factor(i) < 80
            plan_factor(i)  = 85;
        elseif plan_factor(i)>= 80 && plan_factor(i) < 90
            plan_factor(i)  = 90;
        elseif plan_factor(i) >= 90
         if i < numel(plan_factor)  
                continue; 
            else   
            return;
        end
    end
end