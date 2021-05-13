$(document).ready(function() {
	$(chart_id).highcharts({
		chart: chart,
		title: title,
		xAxis: xAxis,
		yAxis: yAxis,
		series:[{name: 'Minimum Temperature', data: data,  pointStart: Date.UTC(2015, 0, 1) , pointInterval: 24 * 3600 * 1000 }],
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


