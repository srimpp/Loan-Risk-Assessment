// 색상 데이터
let color1 = ["#f3a683", '#f7d794', '#778beb', '#e77f67', '#cf6a87', '#786fa6',
	'#f8a5c2', '#63cdda', '#ea8685', '#596275']
let color2 = ['#fc5c65', '#fd9644', '#fed330', '#26de81', '#2bcbba', '#45aaf2',
	'#4b7bec', '#a55eea', '#d1d8e0', '#778ca3']

function a() {
	//변수 데이터
	// 수입
	var income = document.getElementsByClassName("income");
	const ctx_income = document.getElementById("chart_income");

	// 연령
	var ctx_age = document.getElementById("chart_age");
	var age = document.getElementsByClassName("age");

	// 경력
	var ctx_exp = document.getElementById("chart_exp");
	var exp = document.getElementsByClassName("experience");

	// 결혼 여부
	const ctx_marry = document.getElementById("chart_marry");
	var marry = document.getElementsByClassName("marry");

	// 주택 소유 여부
	const ctx_house = document.getElementById("chart_house");
	var house = document.getElementsByClassName("house");

	// 주
	var ctx_state = document.getElementById("chart_state");
	var state1 = document.getElementById("state1").value;
	var state2 = document.getElementById("state2").value;
	var state3 = document.getElementById("state3").value;
	var state4 = document.getElementById("state4").value;
	var state5 = document.getElementById("state5").value;

	// 도시
	var ctx_city = document.getElementById("chart_city");
	var city1 = document.getElementById("city1").value;
	var city2 = document.getElementById("city2").value;
	var city3 = document.getElementById("city3").value;
	var city4 = document.getElementById("city4").value;
	var city5 = document.getElementById("city5").value;
	var city6 = document.getElementById("city6").value;
	var city7 = document.getElementById("city7").value;
	var city8 = document.getElementById("city8").value;
	var city9 = document.getElementById("city9").value;
	var city10 = document.getElementById("city10").value;

	// 근속년수
	var ctx_cur_job = document.getElementById("chart_cur_job");
	var Cur_Job_1 = document.getElementById("Cur_Job_1").value;
	var Cur_Job_2 = document.getElementById("Cur_Job_2").value;
	var Cur_Job_3 = document.getElementById("Cur_Job_3").value;

	// 거주년수
	var ctx_cur_house = document.getElementById("chart_cur_house");
	var Cur_House_1 = document.getElementById("Cur_House_1").value;
	var Cur_House_2 = document.getElementById("Cur_House_2").value;
	var Cur_House_3 = document.getElementById("Cur_House_3").value;
	var Cur_House_4 = document.getElementById("Cur_House_3").value;
	var Cur_House_5 = document.getElementById("Cur_House_3").value;

	// 위험도
	var ctx_risk = document.getElementById("chart_risk");
	var Risk_Flag_safe = document.getElementById("Risk_Flag_safe").value;
	var Risk_Flag_warning = document.getElementById("Risk_Flag_warning").value;

	// 차량 소유
	const ctx_car = document.getElementById("chart_car");
	var car_yes = document.getElementById("car_yes").value;
	var car_no = document.getElementById("car_no").value;

	// 산업군
	var ctx_prof = document.getElementById("chart_prof");
	const var_1 = document.getElementById("prof_1").value;
	const var_2 = document.getElementById("prof_2").value;
	const var_3 = document.getElementById("prof_3").value;
	const var_4 = document.getElementById("prof_4").value;
	const var_5 = document.getElementById("prof_5").value;
	const var_6 = document.getElementById("prof_6").value;
	const var_7 = document.getElementById("prof_7").value;
	const var_8 = document.getElementById("prof_8").value;
	const var_9 = document.getElementById("prof_9").value;
	const var_10 = document.getElementById("prof_10").value;
	const var_11 = document.getElementById("prof_11").value;
	const var_12 = document.getElementById("prof_12").value;
	const var_13 = document.getElementById("prof_13").value;
	const var_14 = document.getElementById("prof_14").value;
	const var_15 = document.getElementById("prof_15").value;


	// 수입
	let chart_income = new Chart(ctx_income, {
		type: 'doughnut',
		data: {
			labels: ['상위권', '중위권', '하위권'],
			datasets: [{
				label: '소득 수준 비율',
				data: [income[0].value, income[1].value, income[2].value],
				backgroundColor: [
					color2[0], color2[1], color2[2]
				],
				hoverOffset: 4,
			}]
		},
		options: {
			title: {
				text: "소득분위 분포(명)",
				display: true
			}
		}
	});


	// 연령
	var chart_age = new Chart(ctx_age, {
		type: "bar",
		data: {
			labels: ['20대', '30대', '40대', '50대', '60대', '70대'],
			datasets: [
				{
					label: "age",
					backgroundColor: ["#f3a683", '#f7d794', '#778beb', '#e77f67', '#cf6a87', '#786fa6'],
					borderColor: "#417690",
					data: [age[0].value, age[1].value, age[2].value, age[3].value,
					age[4].value, age[5].value]
				}
			]
		},
		options: {
			legend: { display: false },
			title: {
				text: "연령대별 인구(명)",
				display: true
			}
		}
	});


	// 경력
	var chart_exp = new Chart(ctx_exp, {
		type: "bar",
		data: {
			labels: ['0-5년차', '5-10년차', '10-15년차', '15-20년차'],
			datasets: [
				{
					label: '전체 산업군의 경력 분포(명)',
					backgroundColor: [color2[0], color2[1], color2[2]],
					borderColor: '#417690',
					data: [exp[0].value, exp[1].value, exp[2].value, exp[3].value]
				}
			]
		},
		options: {
			title: {
				text: "경력별 인구(명)",
				display: true
			}
		}
	});


	// 결혼 여부
	var chart_marry = new Chart(ctx_marry, {
		type: 'pie',
		data: {
			labels: ['기혼', '미혼'],
			datasets: [{
				label: 'House_Ownership',
				data: [marry[0].value, marry[1].value],
				backgroundColor: [
					'rgb(255, 99, 132)',
					'rgb(54, 162, 235)',
					'rgb(255, 205, 86)'
				],
				hoverOffset: 4
			}]
		},
		options: {
			title: {
				text: "결혼 여부",
				display: true
			}
		}
	});


	// 집 소유 종류 파이
	var chart_house = new Chart(ctx_house, {
		type: 'pie',
		data: {
			labels: ['집 소유', '집 렌트', '둘다 아님'],
			datasets: [{
				label: 'House_Ownership',
				data: [house[2].value, house[0].value, house[1].value,],
				backgroundColor: [
					'rgb(255, 99, 132)',
					'rgb(54, 162, 235)',
					'rgb(255, 205, 86)'
				],
				hoverOffset: 4
			}]
		},
		options: {
			title: {
				text: "집 소유 여부",
				display: true
			}
		}
	});


	// STATE(주)
	var chart_state = new Chart(ctx_state, {
		type: "bar",
		data: {
			labels: ["Uttar_Pradesh", "Maharashtra", "Andhra_Pradesh", "West_Bengal", "Bihar"],
			datasets: [
				{
					label: "주별 인구수(명)",
					backgroundColor: [color2[0], color2[1], color2[2], color2[3], color2[4]],
					borderColor: "#417690",
					data: [state1, state2, state3, state4, state5]
				}
			]
		},
		options: {
			title: {
				text: "최다인구 주 Top5",
				display: true
			}
		}
	});


	// 도시
	var chart_city = new Chart(ctx_city, {
		type: "bar",
		data: {
			labels: ["Vijayanagaram", "Bhopal", "Bulandshahr", "Saharsa_29", "Vijayawada", "Srinagar", "Indore", "Hajipur_31", "New_Delhi", "Satara"],
			datasets: [
				{
					label: "도시(명)",
					backgroundColor: [color2[0], color2[1], color2[2], color2[3], color2[4], color2[5], color2[6], color2[7], color2[8], color2[9]],
					borderColor: "#417690",
					data: [city1, city2, city3, city4, city5, city6, city7, city8, city9, city10]
				}
			]
		},
		options: {
			title: {
				text: "최다인구 도시 Top10",
				display: true
			}
		}
	});


	// 근속년수
	var chart_cur_job = new Chart(ctx_cur_job, {
		type: "bar",
		data: {
			labels: ["5년이하", "5년이상-10년이하", "15년이하"],
			datasets: [
				{
					label: "근속기간(년)",
					backgroundColor: [color2[0], color2[1], color2[2]],
					borderColor: "#417690",
					data: [Cur_Job_1, Cur_Job_2, Cur_Job_3]
				}
			]
		},
		options: {
			title: {
				text: "근속년수_Bar",
				display: true
			}
		}
	});


	// 거주년수
	var chart_cur_house = new Chart(ctx_cur_house, {
		type: "bar",
		data: {
			labels: ["10년", "11년", "12년", "13년", "14년"],
			datasets: [
				{
					label: "거주기간(년)",
					backgroundColor: [color2[0], color2[1], color2[2], color2[3], color2[4]],
					borderColor: "#417690",
					data: [Cur_House_1, Cur_House_2, Cur_House_3, Cur_House_4, Cur_House_5]
				}
			]
		},
		options: {
			title: {
				text: "거주년도_Bar",
				display: true
			}
		}
	});


	// 위험도
	var chart_risk = new Chart(ctx_risk, {
		type: "bar",
		data: {
			labels: ["안전", "위험"],
			datasets: [
				{
					label: "위험도",
					backgroundColor: [color2[0], color2[1]],
					borderColor: "#417690",
					data: [Risk_Flag_safe, Risk_Flag_warning]
				}
			]
		},
		options: {
			title: {
				text: "이게 바로 chart_bar입니다.",
				display: true
			}
		}
	});


	// 차량소유 여부
	var chart_car = new Chart(ctx_car, {
		type: 'pie',
		data: {
			labels: [
				'차량 소유',
				'차량 미소유',
			],
			datasets: [{
				label: 'My First Dataset',
				data: [car_yes, car_no],
				backgroundColor: [
					'rgb(255, 99, 132)',
					'rgb(54, 162, 235)',
				],
				hoverOffset: 4
			}]
		}
	});



	var chart_prof = new Chart(ctx_prof, {
		type: 'pie',
		data: {
			labels: [
				'Architecture_Engineering',
				'Arts_Design_Entertainment_Sports_Media',
				'Business_Financial',
				'Computer_Mathematical',
				'Education_Training_Library',
				'Food_Preparation_Serving_Related',
				'Healthcare_Practitioners_Technical',
				'Legal',
				'Life_Physical_Social_Science',
				'Management',
				'Military_Specific',
				'Office_Administrative_Support',
				'Personal_Care_Service',
				'Protective_Service',
				'Transportation_Material_Moving'
			],
			datasets: [{
				label: '직업 대분류별 인원',
				data: [var_1, var_2, var_3, var_4, var_5, var_6,
					var_7, var_8, var_9, var_10, var_11, var_12,
					var_13, var_14, var_15],
				backgroundColor: [
					// 기본적으로 파스텔 색상을 지향함.
					'rgb(255, 99, 132)', // 빨강
					'rgb(54, 162, 235)', // 파랑
					'rgb(255, 205, 86)', // 노랑
					'rgb(243, 166, 131)', // 카라멜 피치
					'rgb(87, 75, 144)', // 퍼플 코랄 라이트
					'rgb(247, 143, 179)', // 플라밍고 핑크
					'rgb(61, 193, 211)', // 블루 쿠라카오?
					'rgb(230, 103, 103)', // 포르세라인 로즈
					'rgb(48, 57, 82)', // 비스케이
					'rgb(85, 230, 193)', // 민트
					'rgb(202, 211, 200)', // 회색
					'rgb(194, 247, 132)', // 메로나
					'rgb(147, 181, 198)', // 죠스바
					'rgb(232, 246, 239)', // 물빠진 청바지
					'rgb(134, 122, 233)' // 아이폰12 퍼플색
				],
				hoverOffset: 4
			}]
		}
	});
}