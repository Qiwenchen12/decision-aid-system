% 提取列的数据
light_factor = table8.light_factor;
temperature_factor = table8.temperature_factor;
humidity_factor = table8.humidity_factor;
wind_factor = table8.wind_factor;
noise_factor = table8.noise_factor;
PM10_factor = table8.PM10_factor;
PM25_factor = table8.PM25_factor;
CO2_factor = table8.CO2_factor;
CO_factor = table8.CO_factor;
TVOC_factor = table8.TVOC_factor;
flowdensity_factor = table8.flowdensity_factor;
natural_factor = table8.natural_factor;
social_factor = table8.social_factor;
employee_factor = table8.employee_factor;
technical_factor = table8.technical_factor;
skill_factor = table8.skill_factor;
basiclocation_factor = table8.basiclocation_factor;
emergencylocation_factor = table8.emergencylocation_factor;
basicalconfiguration_factor = table8.basicalconfiguration_factor;
emergencyconfiguration_factor = table8.emergencyconfiguration_factor;
repair_factor = table8.repair_factor;
qualification_factor = table8.qualification_factor;
safety_factor = table8.safety_factor;
drill_factor = table8.drill_factor;
management_factor = table8.management_factor;
plan_factor = table8.plan_factor;
inspection_factor = table8.inspection_factor;
risk_factor = table8.risk_factor;
organization_factor = table8.organization_factor;
bacteria_factor = table8.bacteria_factor;


% 调用函数计算调整列
social_factor = adjust_social_factor(social_factor);
plan_factor= adjust_plan_factor(plan_factor);
employee_factor =adjust_employee_factor (employee_factor);
PM10_factor= adjust_PM10_factor(PM10_factor);
management_factor= adjust_management_factor(management_factor);
bacteria_factor= adjust_bacteria_factor(bacteria_factor);
PM25_factor= adjust_PM25_factor (PM25_factor);
inspection_factor = adjust_inspection_factor(inspection_factor);
CO_factor= adjust_CO_factor(CO_factor);
technical_factor= adjust_technical_factor(technical_factor);
qualification_factor= adjust_qualification_factor(qualification_factor);
light_factor= adjust_light_factor(light_factor);
drill_factor= adjust_drill_factor(drill_factor);
CO2_factor = adjust_CO2_factor(CO2_factor);
risk_factor = adjust_risk_factor(risk_factor);
emergencyconfiguration_factor= adjust_emergencyconfiguration_factor(emergencyconfiguration_factor);
basiclocation_factor= adjust_basiclocation_factor(basiclocation_factor);
emergencylocation_factor= adjust_emergencylocation_factor(emergencylocation_factor);
noise_factor= adjust_noise_factor(noise_factor);
natural_factor =adjust_natural_factor (natural_factor);
TVOC_factor = adjust_TVOC_factor(TVOC_factor);
organization_factor= adjust_organization_factor(organization_factor);
skill_factor= adjust_skill_factor(skill_factor);
safety_factor= adjust_safety_factor(safety_factor);
repair_factor= adjust_repair_factor(repair_factor);
humidity_factor= adjust_humidity_factor(humidity_factor);
basicalconfiguration_factor= adjust_basicalconfiguration_factor(basicalconfiguration_factor);
flowdensity_factor= adjust_flowdensity_factor(flowdensity_factor);
temperature_factor= adjust_temperature_factor(temperature_factor);
wind_factor= wind_factor;

% 创建一个新表格
adjustedTable1 = table8;
adjustedTable1.employee_factor = employee_factor;

adjustedTable2 = table8;
adjustedTable2.safety_factor = safety_factor;

adjustedTable3 = table8;
adjustedTable3.PM25_factor = PM25_factor;

adjustedTable4 = table8;
adjustedTable4.PM10_factor = PM10_factor;

adjustedTable5 = table8;
adjustedTable5.emergencyconfiguration_factor = emergencyconfiguration_factor;

adjustedTable6 = table8;
adjustedTable6.bacteria_factor = bacteria_factor;

adjustedTable7 = table8;
adjustedTable7.drill_factor = drill_factor;

adjustedTable8 = table8;
adjustedTable8.organization_factor = organization_factor;

adjustedTable9 = table8;
adjustedTable9.qualification_factor = qualification_factor;

adjustedTable10 = table8;
adjustedTable10.technical_factor = technical_factor;

adjustedTable11 = table8;
adjustedTable11.plan_factor = plan_factor;

adjustedTable12 = table8;
adjustedTable12.CO2_factor = CO2_factor;

adjustedTable13 = table8;
adjustedTable13.repair_factor = repair_factor;

adjustedTable14 = table8;
adjustedTable14.social_factor = social_factor;

adjustedTable15 = table8;
adjustedTable15.basiclocation_factor = basiclocation_factor;

adjustedTable16 = table8;
adjustedTable16.skill_factor = skill_factor;

adjustedTable17 = table8;
adjustedTable17.temperature_factor = temperature_factor;

adjustedTable18 = table8;
adjustedTable18.natural_factor = natural_factor;

adjustedTable19 = table8;
adjustedTable19.flowdensity_factor = flowdensity_factor;

adjustedTable20 = table8;
adjustedTable20.management_factor = management_factor;

adjustedTable21 = table8;
adjustedTable21.light_factor = light_factor;

adjustedTable22 = table8;
adjustedTable22.CO_factor = CO_factor;

adjustedTable23 = table8;
adjustedTable23.basicalconfiguration_factor = basicalconfiguration_factor;

adjustedTable24 = table8;
adjustedTable24.inspection_factor = inspection_factor;

adjustedTable25 = table8;
adjustedTable25.risk_factor = risk_factor;

adjustedTable26 = table8;
adjustedTable26.noise_factor = noise_factor;

adjustedTable27 = table8;
adjustedTable27.TVOC_factor = TVOC_factor;

adjustedTable28 = table8;
adjustedTable28.wind_factor = wind_factor;

adjustedTable29 = table8;
adjustedTable29.humidity_factor = humidity_factor;

adjustedTable30 = table8;
adjustedTable30.emergencylocation_factor = emergencylocation_factor;





%放在同一表下
table9 = vertcat(adjustedTable1, adjustedTable2,adjustedTable3,adjustedTable4,adjustedTable5,adjustedTable6,adjustedTable7,adjustedTable8,adjustedTable9,adjustedTable10,adjustedTable11, adjustedTable12,adjustedTable13,adjustedTable14,adjustedTable15,adjustedTable16,adjustedTable17,adjustedTable18,adjustedTable19,adjustedTable20,adjustedTable21, adjustedTable22,adjustedTable23,adjustedTable24,adjustedTable25,adjustedTable26,adjustedTable27,adjustedTable28,adjustedTable29,adjustedTable30);


%删除重复项
[~, uniqueIndices] = unique(table9, 'rows', 'stable');
table9Unique = table9(uniqueIndices, :);

rows_table8Unique_noDup = table8{:,:};
rows_table9Unique = table9Unique{:,:};
[~, idx1] = ismember(rows_table9Unique, rows_table8Unique_noDup, 'rows');
table9 = table9Unique(~idx1, :);

