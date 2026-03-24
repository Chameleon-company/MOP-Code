#create and switch to database schema
create schema melb_open_data;
use melb_open_data;

#create tables for all datasets being used in project so far
#then import data for each table
#sample data and add indexes

create table pedestrian_data as select * from mydb.test_1;

create table population_data (year varchar(255), residents	varchar(255), workers varchar(255),
    students varchar(255),metropolitan_visitors varchar(255), regional_visitors	varchar(255),
    interstate_visitors varchar(255),	international_visitors varchar(255), under_15_years varchar(255),
    total_population varchar(255));


load data infile '/Users/hannahcatri/capstone-melbourne/T3_2020/Dashboards/populations_historic_and_predictions.csv'
    into table population_data
    fields terminated by ','
    lines terminated by '\r\n'
    ignore 1 lines;

select * from population_data limit 100;
select * from pedestrian_data limit 100;

alter table pedestrian_data modify Date varchar(255);
alter table pedestrian_data modify Time varchar(255);
alter table pedestrian_data add index(sensor_id, Date,Time);

#this one view is being created for pedestrians dashboard
create view relativePedestianCounts as (select ped.year, sum(ped.hourly_counts) as total_pedestrians,
        (pop.total_population*1000) as 'total_population',
       format(sum(ped.hourly_counts)/(pop.total_population*1000),0) as 'count_perCapita'
    from pedestrian_data ped
    join population_data pop
    on ped.year=pop.year
    group by ped.year);

drop table if exists pedestrian_microclimate;
create table pedestrian_microclimate (Date varchar(255), Day_of_week varchar(255), Week	varchar(255), Day varchar(255),	Average_Temp_Max varchar(255),	Average_Temp_Min varchar(255),
    Average_Humidity_Perc varchar(255),	Average_Wind_Speed_KMhr varchar(255),	Pedestrian_Total_Count_Daily varchar(255),	Pedestrian_Day_Avg varchar(255));

load data infile '/Users/hannahcatri/capstone-melbourne/T3_2020/Dashboards/Pedestrian_micro_climate_day_avg_11_2019_till_12_2020.csv'
    into table pedestrian_microclimate
    fields terminated by ','
    lines terminated by '\r\n'
    ignore 1 lines;

select * from pedestrian_microclimate limit 100;
alter table pedestrian_microclimate add index(Date);

create table heatmap_TidyPedL (date_time varchar(255), sensor_id varchar(255), hourly_counts varchar(255), sensor_description varchar(255),	latitude varchar(255),	longitude varchar(255));


load data infile '/Users/hannahcatri/capstone-melbourne/D2I - Melbourne City_2020T3/T3_2020/Heatmap/TidyPedL.csv'
    into table heatmap_TidyPedL
    fields terminated by ','
    lines terminated by '\r\n'
    ignore 1 lines;

alter table heatmap_TidyPedL add index(date_time,sensor_id);

drop table if exists pedestrian_with_public_holidays;
create table pedestrian_with_public_holidays (date varchar(255), day varchar(255), monthly_index varchar(255), covid_restrictions varchar(255),	rainfall_amount_mm varchar(255), minimum_temperature_degree_c varchar(255),
    maximum_temperature_degree_c varchar(255),	daily_global_solar_exposure varchar(255),total_pedestrian_count_per_day varchar(255));

load data infile '/Users/hannahcatri/capstone-melbourne/D2I - Melbourne City_2020T3/T3_2020/dataset_with_public_holidays.csv'
    into table pedestrian_with_public_holidays
    fields terminated by ','
    lines terminated by '\r\n'
    ignore 1 lines;
