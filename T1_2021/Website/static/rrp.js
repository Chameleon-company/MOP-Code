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

$(document).ready(function() {
	$(chart_id_4).highcharts({
		chart: chart4,
		title: title4,
		xAxis: xAxis4,
		yAxis: yAxis4,
		series: series4,
		tooltip: {headerFormat: '{point.key:%b\'%y}<br/>', shared: true ,         formatter: function () {
            return 'Demand of <b>' + this.y +
                '  </b>MWh has RRP of <b>' + this.x + '</b> dollars';
        }},
	});
});


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
	$(chart_id_6).highcharts({
		chart: chart6,
		title: title6,
		xAxis: xAxis6,
		yAxis: yAxis6,
		series: series6,
		tooltip: {headerFormat: '{point.key:%b\'%y}<br/>', shared: true ,         formatter: function () {
            return 'Demand of <b>' + this.y +
                '  </b>MWh has RRP of <b>' + this.x + '</b> dollars';

        }},
	});
});


$(document).ready(function() {
	$(chart_id_6).highcharts({
		chart: chart6,
		title: title6,
		xAxis: xAxis6,
		yAxis: yAxis6,
		series: series6,
		tooltip: {headerFormat: '{point.key:%b\'%y}<br/>', shared: true ,         formatter: function () {
            return 'Demand of <b>' + this.y +
                '  </b>MWh has RRP of <b>' + this.x + '</b> dollars';

        }},
	});
});


$(document).ready(function() {
	$(chart_id_7).highcharts({
		chart: chart7,
		title: title7,
		xAxis: xAxis7,
		yAxis: yAxis7,
		series:[{name: 'Actual RRP', data: data7,  pointStart: Date.UTC(2015, 0, 1) , pointInterval: 24 * 3600 * 1000 },{name: 'Predicted RRP', data: data7_2,  pointStart: Date.UTC(2021, 2, 1) , pointInterval: 24 * 3600 * 1000 }],
	});
})