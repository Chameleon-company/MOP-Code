$(document).ready(function() {
	$(chart_id).highcharts({
		chart: chart,
		title: title,
		xAxis: xAxis,
		yAxis: yAxis,
		series:[{name: 'Maximum Temperature', data: data,  pointStart: Date.UTC(2015, 0, 1) , pointInterval: 24 * 3600 * 1000 }],
		tooltip: {pointFormat: 'x: <b>{point.x:%d/%m/%y}</b><br>y: <b>{point.y}</b>' }
	});
})


$(document).ready(function() {
	$(chart_id_5).highcharts({
		chart: chart5,
		title: title5,
		xAxis: xAxis5,
		yAxis: yAxis5,
		series: series5,
		tooltip: {headerFormat: '{point.key:%b\'%y}<br/>', shared: true ,         formatter: function () {
            return 'The frequency of RRP between the range of <b>' + this.x +
                '  </b> dollars is <b>' + this.y + '</b>';

        }},
	});
});

$(document).ready(function() {
	$(chart_id_2).highcharts({
		chart: chart2,
		title: title2,
		xAxis: xAxis2,
		yAxis: yAxis2,
		series: [{        name: 'Observations',        data: [
				[16.7,21.8,25.3,30.3,42.9],
				[17.3,21.2,24.6,29.3,39.4],
				[15.3,20.5,23.3,27.4,36.3],
				[14.5,18.1,20.2,23.9,31.6],
				[10.6,15.325, 17.1,19.5,25.7],
				[9.6,13.275,14.5,16.025,19.5],
				[9.8,12.925,14.3,15.575,19.5],
				[9.0,13.0,14.8,16.4,20.7],
				[11.9,14.9,16.7,19.625,26.7125],
				[12.6,16.725,19.5,24.9,35.8],
				[14.2,18.1,20.7,28.125,40.9,],
				[15.5,20.625,23.55,28.375,40.1]
        ],
		
        tooltip: { headerFormat: '<em>Experiment No {point.key}</em><br/>'      }
    }, {
        name: 'Outliers',
        color: Highcharts.getOptions().colors[0],
        type: 'scatter',
        data: [ // x, y positions where 0 is the first category
				[2, 38.1],
				[2, 38.9],
				[7, 21.6],
				[8, 30.6],
				[11, 40.8],
				[11, 41.2],
				[11, 43.5],
        ],
        marker: {
            fillColor: 'white',
            lineWidth: 1,
            lineColor: Highcharts.getOptions().colors[0]
        },
        tooltip: {pointFormat: 'Observation: {point.y}'}
    }],
	});
});


$(document).ready(function() {
	$(chart_id_7).highcharts({
		chart: chart7,
		title: title7,
		xAxis: xAxis7,
		yAxis: yAxis7,
		series:[{name: 'Actual Maximum Temperature', data: data,  pointStart: Date.UTC(2015, 0, 1) , pointInterval: 24 * 3600 * 1000 },{name: 'Predicted Maximum Temperature', data: data7_2,  pointStart: Date.UTC(2021, 2, 1) , pointInterval: 24 * 3600 * 1000 }],
		tooltip: {pointFormat: 'x: <b>{point.x:%d/%m/%y}</b><br>y: <b>{point.y}</b>' }		
	});
})