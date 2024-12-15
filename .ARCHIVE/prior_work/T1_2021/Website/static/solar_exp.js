$(document).ready(function() {
	$(chart_id).highcharts({
		chart: chart,
		title: title,
		xAxis: xAxis,
		yAxis: yAxis,
		series:[{name: 'Solar exposure', data: data,  pointStart: Date.UTC(2015, 0, 1) , pointInterval: 24 * 3600 * 1000 }],
		tooltip: {pointFormat: 'x: <b>{point.x:%d/%m/%y}</b><br>y: <b>{point.y}</b>' }
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
            return 'Mean solar exposure for <b>' + this.x +
                '</b> is <b>' + this.y + 'mm</b>';
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
            return 'Solar exposure amount for <b>' + this.x +
                '</b> is <b>' + this.y + 'mm</b>';
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
            return 'Solar exposure amount for <b>' + this.x +
                '</b> is <b>' + this.y + 'mm</b>';
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
            return 'The frequency of solar exposure between <b>' + this.x +
                ' MJ/m*m range</b> is <b>' + this.y + '</b>';

        }},
	});
});

$(document).ready(function() {
	$(chart_id_7).highcharts({
		chart: chart7,
		title: title7,
		xAxis: xAxis7,
		yAxis: yAxis7,
		series:[{name: 'Actual Solar exp', data: data,  pointStart: Date.UTC(2015, 0, 1) , pointInterval: 24 * 3600 * 1000 },{name: 'Predicted solar exp', data: data7_2,  pointStart: Date.UTC(2021, 2, 1) , pointInterval: 24 * 3600 * 1000 }],
		tooltip: {pointFormat: 'x: <b>{point.x:%d/%m/%y}</b><br>y: <b>{point.y}</b>' }		
	});
})

