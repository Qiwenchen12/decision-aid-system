


% 读取数据
% 给定的一行数据
data_row =  [50, 30.5, 68, 0.9, 68, 220, 110, 0.23, 5.7, 0.48,140, 0.85, 7.8, 0.2, 3, 85, 72, 50, 60, 90,75, 70, 95, 12, 6, 50, 60, 0.2, 0.4, 50];

% 将一行数据转换为table
column_names = {'light_factor', 'temperature_factor', 'humidity_factor', 'wind_factor', 'noise_factor', 'PM10_factor', 'PM25_factor', 'CO2_factor', 'CO_factor', 'TVOC_factor', ...
    'bacteria_factor', 'flowdensity_factor', 'natural_factor', 'social_factor', 'employee_factor', 'technical_factor', 'skill_factor', 'basiclocation_factor', 'emergencylocation_factor', 'basicalconfiguration_factor', ...
    'emergencyconfiguration_factor', 'repair_factor', 'qualification_factor', 'safety_factor', 'drill_factor', 'management_factor', 'plan_factor', 'inspection_factor', 'risk_factor', 'organization_factor'};

data_table = array2table(data_row, 'VariableNames', column_names);

light_factor = data_table.light_factor;
temperature_factor = data_table.temperature_factor;
humidity_factor = data_table.humidity_factor;
wind_factor = data_table.wind_factor;
noise_factor = data_table.noise_factor;
PM10_factor = data_table.PM10_factor;
PM25_factor = data_table.PM25_factor;
CO2_factor = data_table.CO2_factor;
CO_factor = data_table.CO_factor;
TVOC_factor = data_table.TVOC_factor;
bacteria_factor = data_table.bacteria_factor;
flowdensity_factor = data_table.flowdensity_factor;
natural_factor = data_table.natural_factor;
social_factor = data_table.social_factor;
employee_factor = data_table.employee_factor;
technical_factor = data_table.technical_factor;
skill_factor = data_table.skill_factor;
basiclocation_factor = data_table.basiclocation_factor;
emergencylocation_factor = data_table.emergencylocation_factor;
basicalconfiguration_factor = data_table.basicalconfiguration_factor;
emergencyconfiguration_factor = data_table.emergencyconfiguration_factor;
repair_factor = data_table.repair_factor;
qualification_factor = data_table.qualification_factor;
safety_factor = data_table.safety_factor;
drill_factor = data_table.drill_factor;
management_factor = data_table.management_factor;
plan_factor = data_table.plan_factor;
inspection_factor = data_table.inspection_factor;
risk_factor = data_table.risk_factor;
organization_factor = data_table.organization_factor;




% 将所有V级调节为IV
if light_factor < 60
    light_factor = 65; 
else 
    light_factor = light_factor;
end

if temperature_factor > 30
    temperature_factor = 29; 
else 
    temperature_factor = temperature_factor;
end

if humidity_factor >= 80
    humidity_factor = 75; 
else 
    humidity_factor = humidity_factor;
end

if wind_factor >= 1
    wind_factor = 0.75; 
else 
    wind_factor = wind_factor;
end

if noise_factor >= 85
    noise_factor = 80; 
else 
    noise_factor = noise_factor;
end

if PM10_factor >= 350
    PM10_factor = 300; 
else 
    PM10_factor = PM10_factor;
end

if PM25_factor >= 150
    PM25_factor = 132.5; 
else 
    PM25_factor = PM25_factor;
end

if CO2_factor >= 0.25
    CO2_factor = 0.2; 
else 
    CO2_factor = CO2_factor;
end

if CO_factor >= 15
    CO_factor = 12.5; 
else 
    CO_factor = CO_factor;
end


if TVOC_factor >= 0.8
    TVOC_factor = 0.7; 
else 
    TVOC_factor = TVOC_factor;
end

if bacteria_factor >= 7000
    bacteria_factor = 5500; 
else 
    bacteria_factor = bacteria_factor;
end

if flowdensity_factor >= 4.18
    flowdensity_factor = 3.72; 
else 
    flowdensity_factor = flowdensity_factor;
end

if natural_factor >= 8
    natural_factor = 7; 
else 
    natural_factor = natural_factor;
end

if social_factor == 0.2
    social_factor = 0.4;  
else 
    social_factor = social_factor;
end


if employee_factor <= 3
    employee_factor = 4; 
else 
    employee_factor = employee_factor;
end


if technical_factor <= 80
    technical_factor = 82.5; 
else 
    technical_factor = technical_factor;
end


if skill_factor <= 60
    skill_factor = 65; 
else 
    skill_factor = skill_factor;
end

if basiclocation_factor <= 60
    basiclocation_factor = 65; 
else 
    basiclocation_factor = basiclocation_factor;
end

if emergencylocation_factor <= 60
    emergencylocation_factor = 65; 
else 
    emergencylocation_factor = emergencylocation_factor;
end

if basicalconfiguration_factor <= 80
    basicalconfiguration_factor = 82.5; 
else 
    basicalconfiguration_factor = basicalconfiguration_factor;
end


if emergencyconfiguration_factor <= 80
    emergencyconfiguration_factor = 82.5; 
else 
    emergencyconfiguration_factor = emergencyconfiguration_factor;
end

if repair_factor <= 60
    repair_factor = 65; 
else 
    repair_factor = repair_factor;
end

if qualification_factor <= 80
    qualification_factor = 82.5; 
else 
    qualification_factor = qualification_factor;
end

if safety_factor <= 12
    safety_factor = 18; 
else 
    safety_factor = safety_factor;
end

if drill_factor <= 6
    drill_factor = 9; 
else 
    drill_factor = drill_factor;
end

if management_factor <= 60
    management_factor = 65; 
else 
    management_factor = management_factor;
end

if plan_factor <= 60
    plan_factor = 65; 
else 
    plan_factor = plan_factor;
end

if inspection_factor == 0.2
    inspection_factor = 0.4;  
else 
    inspection_factor = inspection_factor;
end

if risk_factor == 0.2
    risk_factor = 0.4;  
else 
    risk_factor = risk_factor;
end

if organization_factor <= 60
    organization_factor = 65; 
else 
    organization_factor = organization_factor;
end

% data0导入数据表
table0 = table(light_factor, temperature_factor, humidity_factor, wind_factor, noise_factor, PM10_factor, PM25_factor, CO2_factor, CO_factor, TVOC_factor, bacteria_factor, flowdensity_factor, natural_factor, social_factor, employee_factor, technical_factor, skill_factor, basiclocation_factor, emergencylocation_factor, basicalconfiguration_factor, emergencyconfiguration_factor, repair_factor, qualification_factor, safety_factor, drill_factor, management_factor, plan_factor, inspection_factor, risk_factor, organization_factor);

