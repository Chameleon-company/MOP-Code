$(document).ready(function() {
	$(chart_id).highcharts({
		chart: chart,
		title: title,
		xAxis: xAxis,
		yAxis: yAxis,
		series: series,
		tooltip: {headerFormat: '{point.key:%b\'%y}<br/>', shared: true,          formatter: function () {
            return 'Rainfall amount for <b>' + this.x +
                '</b> is <b>' + this.y + 'mm</b>';
        } },
	});
});

$(document).ready(function() {
	$(chart_id_2).highcharts({
		chart: chart2,
		title: title2,
		xAxis: xAxis2,
		yAxis: yAxis2,
		series: series2,
		tooltip: {headerFormat: '{point.key:%b\'%y}<br/>', shared: true ,         formatter: function () {
            return 'Rainfall amount for <b>' + this.x +
                '</b> is <b>' + this.y + 'mm</b>';
        }},
	});
});