$(document).ready(function() {
	$(chart_id).highcharts({
		chart: chart,
		title: title,
		xAxis: xAxis,
		yAxis: yAxis,
		series:[{name: 'Minimum Temperature', data: data,  pointStart: Date.UTC(2015, 0, 1) , pointInterval: 24 * 3600 * 1000 }],
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
			[10.3,14.4,16.2,18.0,23.1],

            [10.0,13.7,15.8,17.9,23.5],

            [8.5,12.4,14.4,16.6,22.6],

            [6.2,10.15,11.9,13.6,18.6],

            [6,7.8,10.2,12.1,17.1],

            [1.7,5.38,7.6,9.5,13.0],

            [1.8,6.1,7.8,9.18,12.7],

            [0.8,5.8,7.85,9.375,12.8],

            [3.4,7.4,9.0,10.73,15.7],

            [4.2, 9.2, 10.7, 12.9, 17.6],

            [6.4,10.8,12.1,14.8,20],

            [8.8,12.3,14.1,16.375,21.9]
        ],
		
        tooltip: { headerFormat: '<em>Experiment No {point.key}</em><br/>'      }
    }, {
        name: 'Outliers',
        color: Highcharts.getOptions().colors[0],
        type: 'scatter',
        data: [ // x, y positions where 0 is the first category
            [0, 27.8],
			[2, 25.1],
			[3, 19.8],
			[3, 22.1], 
			[6, 1.1],
			[6, 0.6],
			[7, 15.3],
			[8, 17.1],
			[8, 16],
			[8, 16.6],
			[9, 19.5],
			[9, 20],
			[9, 21.7],
			[10, 22.8],
			[10, 21.5],
			[10, 21.9],
			[11, 22.7],
			[11, 24],
			[11, 25.1],
			[11, 25.9],
			[11, 27]
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
		series:[{name: 'Actual Minimum Temperature', data: data,  pointStart: Date.UTC(2015, 0, 1) , pointInterval: 24 * 3600 * 1000 },{name: 'Predicted Minimum Temperature', data: data7_2,  pointStart: Date.UTC(2021, 2, 1) , pointInterval: 24 * 3600 * 1000 }],
		tooltip: {pointFormat: 'x: <b>{point.x:%d/%m/%y}</b><br>y: <b>{point.y}</b>' }		
	});
})

