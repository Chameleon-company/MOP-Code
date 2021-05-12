$(document).ready(function() {
	$(chart_id).highcharts({
		chart: chart,
		title: title,
		xAxis: xAxis,
		yAxis: yAxis,
		series: series,
		tooltip: {headerFormat: '{point.key:%b\'%y}<br/>', shared: true,          formatter: function () {
            return 'Electricity forecast <b>' + this.x +
                '</b> is <b>' + this.y + ' MWH </b>';
        } },
	});
});