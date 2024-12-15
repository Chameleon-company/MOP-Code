DECLARE @minDateTime AS DATETIME;
DECLARE @maxDateTime AS DATETIME;

SET @minDateTime = '2020-01-01 00:00:00'; ----YYYY-MM-DD hh:mm:ss format
SET @maxDateTime = '2020-12-31 23:00:00';

;
WITH Dates_CTE
     AS (SELECT @minDateTime AS Dates
         UNION ALL
         SELECT Dateadd(hh, 1, Dates)
         FROM   Dates_CTE
         WHERE  Dates < @maxDateTime)
SELECT cast(cast(dates as date) as varchar(max))as Date, CONVERT(VARCHAR(5),dates,108) as Time  into  temp_date
FROM   Dates_CTE
OPTION (MAXRECURSION 0)

select * from temp_date
order by date

select * from microclimate_updated

DROP TABLE temp
SELECT * INTO temp FROM(
SELECT A.Date, A.Time, 'Brooklyn' AS Location, B.value FROM temp_date A LEFT JOIN microclimate_updated B ON A.Date = B.Date AND A.Time = B.Time  AND B.location = 'Brooklyn' WHERE A.Date >= (SELECT MIN(Date) FROM microclimate_updated WHERE location = 'Brooklyn') AND A.Date <= (SELECT MAX(Date) FROM microclimate_updated WHERE location = 'Brooklyn')
UNION
SELECT A.Date, A.Time, 'Alphington' AS Location, B.value FROM temp_date A LEFT JOIN microclimate_updated B ON A.Date = B.Date AND A.Time = B.Time  AND B.location = 'Alphington' WHERE A.Date >= (SELECT MIN(Date) FROM microclimate_updated WHERE location = 'Alphington') AND A.Date <= (SELECT MAX(Date) FROM microclimate_updated WHERE location = 'Alphington')
UNION
SELECT A.Date, A.Time, 'Brighton' AS Location, B.value FROM temp_date A LEFT JOIN microclimate_updated B ON A.Date = B.Date AND A.Time = B.Time  AND B.location = 'Brighton' WHERE A.Date >= (SELECT MIN(Date) FROM microclimate_updated WHERE location = 'Brighton') AND A.Date <= (SELECT MAX(Date) FROM microclimate_updated WHERE location = 'Brighton')
UNION
SELECT A.Date, A.Time, 'Coolaroo' AS Location, B.value FROM temp_date A LEFT JOIN microclimate_updated B ON A.Date = B.Date AND A.Time = B.Time  AND B.location = 'Coolaroo' WHERE A.Date >= (SELECT MIN(Date) FROM microclimate_updated WHERE location = 'Coolaroo') AND A.Date <= (SELECT MAX(Date) FROM microclimate_updated WHERE location = 'Coolaroo')
UNION
SELECT A.Date, A.Time, 'Footscray' AS Location, B.value FROM temp_date A LEFT JOIN microclimate_updated B ON A.Date = B.Date AND A.Time = B.Time  AND B.location = 'Footscray' WHERE A.Date >= (SELECT MIN(Date) FROM microclimate_updated WHERE location = 'Footscray') AND A.Date <= (SELECT MAX(Date) FROM microclimate_updated WHERE location = 'Footscray')
UNION
SELECT A.Date, A.Time, 'Dallas' AS Location, B.value FROM temp_date A LEFT JOIN microclimate_updated B ON A.Date = B.Date AND A.Time = B.Time  AND B.location = 'Dallas' WHERE A.Date >= (SELECT MIN(Date) FROM microclimate_updated WHERE location = 'Dallas') AND A.Date <= (SELECT MAX(Date) FROM microclimate_updated WHERE location = 'Dallas')
UNION
SELECT A.Date, A.Time, 'Box Hill' AS Location, B.value FROM temp_date A LEFT JOIN microclimate_updated B ON A.Date = B.Date AND A.Time = B.Time  AND B.location = 'Box Hill' WHERE A.Date >= (SELECT MIN(Date) FROM microclimate_updated WHERE location = 'Box Hill') AND A.Date <= (SELECT MAX(Date) FROM microclimate_updated WHERE location = 'Box Hill')
UNION
SELECT A.Date, A.Time, 'Macleod' AS Location, B.value FROM temp_date A LEFT JOIN microclimate_updated B ON A.Date = B.Date AND A.Time = B.Time  AND B.location = 'Macleod' WHERE A.Date >= (SELECT MIN(Date) FROM microclimate_updated WHERE location = 'Macleod') AND A.Date <= (SELECT MAX(Date) FROM microclimate_updated WHERE location = 'Macleod')
UNION
SELECT A.Date, A.Time, 'Melbourne CBD' AS Location, B.value FROM temp_date A LEFT JOIN microclimate_updated B ON A.Date = B.Date AND A.Time = B.Time  AND B.location = 'Melbourne CBD' WHERE A.Date >= (SELECT MIN(Date) FROM microclimate_updated WHERE location = 'Melbourne CBD') AND A.Date <= (SELECT MAX(Date) FROM microclimate_updated WHERE location = 'Melbourne CBD')
UNION
SELECT A.Date, A.Time, 'Campbellfield' AS Location, B.value FROM temp_date A LEFT JOIN microclimate_updated B ON A.Date = B.Date AND A.Time = B.Time  AND B.location = 'Campbellfield' WHERE A.Date >= (SELECT MIN(Date) FROM microclimate_updated WHERE location = 'Campbellfield') AND A.Date <= (SELECT MAX(Date) FROM microclimate_updated WHERE location = 'Campbellfield')
UNION
SELECT A.Date, A.Time, 'Altona North 2' AS Location, B.value FROM temp_date A LEFT JOIN microclimate_updated B ON A.Date = B.Date AND A.Time = B.Time  AND B.location = 'Altona North 2' WHERE A.Date >= (SELECT MIN(Date) FROM microclimate_updated WHERE location = 'Altona North 2') AND A.Date <= (SELECT MAX(Date) FROM microclimate_updated WHERE location = 'Altona North 2')
UNION
SELECT A.Date, A.Time, 'Altona North 1' AS Location, B.value FROM temp_date A LEFT JOIN microclimate_updated B ON A.Date = B.Date AND A.Time = B.Time  AND B.location = 'Altona North 1' WHERE A.Date >= (SELECT MIN(Date) FROM microclimate_updated WHERE location = 'Altona North 1') AND A.Date <= (SELECT MAX(Date) FROM microclimate_updated WHERE location = 'Altona North 1')
UNION
SELECT A.Date, A.Time, 'Thomastown East' AS Location, B.value FROM temp_date A LEFT JOIN microclimate_updated B ON A.Date = B.Date AND A.Time = B.Time  AND B.location = 'Thomastown East' WHERE A.Date >= (SELECT MIN(Date) FROM microclimate_updated WHERE location = 'Thomastown East') AND A.Date <= (SELECT MAX(Date) FROM microclimate_updated WHERE location = 'Thomastown East')
UNION
SELECT A.Date, A.Time, 'MRI Campbellfield fire' AS Location, B.value FROM temp_date A LEFT JOIN microclimate_updated B ON A.Date = B.Date AND A.Time = B.Time  AND B.location = 'MRI Campbellfield fire' WHERE A.Date >= (SELECT MIN(Date) FROM microclimate_updated WHERE location = 'MRI Campbellfield fire') AND A.Date <= (SELECT MAX(Date) FROM microclimate_updated WHERE location = 'MRI Campbellfield fire')
) as temp

SELECT * FROM temp