// モジュール読み込み
let client = require('cheerio-httpcli');

/**
   スクレイピングする
 */
let scraping = function(shortName) {
    // チームの試合結果のURL
    let url = 'http://www.football-lab.jp' + shortName + 'match/'

    // Promiseを返却する
    return new Promise(function(resolve, reject){
	// スクレイピング
	client.fetch(url, {}, (err, $, res) => {
	    let results = $('#schedule tr').slice(1).map((i, e) => {
		return {
		    date: $(e).find('.tDate').text().trim(),
		    time: $(e).find('.tTime').text().trim(),
		    team: $(e).find('.tTeam').text().trim(),
		    result: $(e).find('.tResult').text().trim(),
		    sta: $(e).find('.tSta').text().trim(),
		};
	    });
	    resolve(results.toArray());
	});
    });
}

client.fetch('http://www.football-lab.jp/summary/team_ranking/j1/?year=2017', {}, function (err, $, res) {
    // チーム一覧を取得する
    let teams = $('#StandingSummary tr').slice(1).map((i, e) => { // slice(1)でヘッダ行を取り除く
	// 略称と正式名称の連想配列を返す
	return {
	    path: $(e).find('.tEmblem a').attr('href'),
	    name: $(e).find('.tName a').text(),
	};
    });
    // チーム一覧に対してスクレイピングを行う
    let resultPromises = teams.map((i, team) => {
	return scraping(team.path);
    }).toArray();

    Promise.all(resultPromises).then((results) => {
	// 出力用に整形する
	let teamMap = {};
	teams.each((i, team) => {
	    teamMap[team.name] = results[i];
	});
	console.log(JSON.stringify(teamMap));
    }, (err) => {
	console.log(err);
    })
});
