$(document).ready(function() {
	$(chart_id).highcharts({
		chart: chart,
		title: title,
		xAxis: xAxis,
		yAxis: yAxis,
		series:[{name: 'Actual Demand', data: data1,  pointStart: Date.UTC(2015, 0, 1) , pointInterval: 24 * 3600 * 1000 },{name: 'Predicted Demand', data: data1_2,  pointStart: Date.UTC(2021, 2, 1) , pointInterval: 24 * 3600 * 1000 }],
		tooltip: {pointFormat: 'x: <b>{point.x:%d/%m/%y}</b><br>y: <b>{point.y}</b>' }
	});
})