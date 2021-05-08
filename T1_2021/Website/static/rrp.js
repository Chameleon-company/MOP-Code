$(document).ready(function() {
	$(chart_id).highcharts({
		chart: chart,
		title: title,
		xAxis: xAxis,
		yAxis: yAxis,
		series: series,
		tooltip: {headerFormat: '{point.key:%b\'%y}<br/>', shared: true,          formatter: function () {
            return 'RRP amount for <b>' + this.x +
                '</b> is <b>' + this.y + ' Dollars </b>';
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
            return 'Average RRP for <b>' + this.x +
                '</b> is <b>' + this.y + ' dollars</b>';
        }},
	});
});

$(document).ready(function() {
	$(chart_id_3).highcharts({
		chart: chart3,
		title: title3,
		xAxis: xAxis3,
		yAxis: yAxis3,
		series: series3,
		tooltip: {headerFormat: '{point.key:%b\'%y}<br/>', shared: true ,         formatter: function () {
            return 'Average RRP for <b>' + this.x +
                '</b> is <b>' + this.y + ' dollars</b>';
        }},
	});
});

// $(document).ready(function() {
	// $(chart_id_4).highcharts({
		// chart: chart4,
		// title: title4,
		// xAxis: xAxis4,
		// yAxis: yAxis4,
		// series: series4,
		// tooltip: {headerFormat: '{point.key:%b\'%y}<br/>', shared: true ,         formatter: function () {
            // return 'Maximum rainfall amount for <b>' + this.x +
                // '</b> is <b>' + this.y + 'mm</b>';
        // }},
	// });
// });


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
