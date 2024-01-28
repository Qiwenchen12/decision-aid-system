function organization_factor = adjust_organization_factor(organization_factor)
for i = 1:numel(organization_factor)
     if organization_factor(i)> 60 &&  organization_factor(i)< 70
            organization_factor(i)= 75;
        elseif organization_factor(i) >= 70 && organization_factor(i) < 80
            organization_factor(i) = 85;
        elseif organization_factor(i)  >= 80 && organization_factor(i) < 90
            organization_factor(i)  = 90;
        elseif organization_factor(i)>= 90
         if i < numel(organization_factor)  
                continue; 
            else   
            return;
        end
    end
end