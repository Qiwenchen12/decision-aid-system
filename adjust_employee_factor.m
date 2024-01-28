function employee_factor = adjust_employee_factor(employee_factor)
for i = 1:numel(employee_factor)
     if employee_factor(i) > 3 &&  employee_factor(i)< 5
            employee_factor(i)= 5.5;
        elseif employee_factor(i) >= 5 && employee_factor(i)  < 6
            employee_factor(i)  = 6.5;
        elseif employee_factor(i) >= 6 && employee_factor(i) < 7
            employee_factor(i)  = 7;
        elseif employee_factor(i) >= 7
         if i < numel(employee_factor)  
                continue; 
            else   
            return;
        end
    end
end